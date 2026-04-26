import argparse
import os
import json
import random
import time
import torchvision

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as FF
import torchvision.transforms.functional as TF
import Transform
import timm

from PIL import Image
from tqdm.auto import tqdm
from torchvision import models, transforms

img_height, img_width = 224, 224


def get_parser():
	parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
	parser.add_argument('--attack', default='raa', type=str, help='the attack algorithm')
	parser.add_argument('--epoch', default=100, type=int, help='the iterations for updating the adversarial patch')
	parser.add_argument('--batchsize', default=12, type=int, help='the bacth size')
	parser.add_argument('--epsilon', default=16 / 255, type=float, help='the stepsize to update the perturbation')
	parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
	parser.add_argument('--momentum', default=0.0, type=float, help='the decay factor for momentum based attack')
	parser.add_argument('--random_start', default=False, type=bool, help='set random start')
	parser.add_argument('--input_dir', default='./data2', type=str, help='the path for custom benign images, default: untargeted attack data')
	parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
	parser.add_argument('--targeted', action='store_true', help='targeted attack')
	parser.add_argument('--GPU_ID', default='cuda:0', type=str)
	parser.add_argument('--max_test_num', default=200, type=int)
	parser.add_argument('--transform', type=str)
	parser.add_argument('--transform_par', type=float)
	parser.add_argument('--save_all', action='store_true', help='save all images in each batch, not just the first')
	parser.add_argument('--n_theta', default=20, type=int, help='the number of theta samples for RAA')
	parser.add_argument('--bb_models', default='resnet34,mobilenet_v3_small,vit_b_16', type=str, help='comma-separated black-box evaluation models')
	parser.add_argument('--en_models', default='vgg19,densenet121,efficientnet_b0,swin_t', type=str, help='comma-separated Ensemble Loss models')
	parser.add_argument('--ensemble', action='store_true', help='Use Ensemble Loss')
	parser.add_argument('--nn', action='store_true', help='Use Disturbance Network')
	parser.add_argument('--lambda1', default=1, type=float, help='weight for randomly sampled model loss')
	parser.add_argument('--lambda2', default=1, type=float, help='weight for mean ensemble loss')
	parser.add_argument('--lambda3', default=0.2, type=float, help='weight for max ensemble loss')
	parser.add_argument('--ensemble_resample_interval', default=2, type=int, help='resample interval for ensemble loss model')
	parser.add_argument('--active_models_k', default=2, type=int, help='number of active models used in optimization internals')
	parser.add_argument('--nn_inner_steps', default=5, type=int, help='DisturbanceNet optimizer steps per outer iteration')
	parser.add_argument('--nn_theta_bank', default=2, type=int, help='number of theta samples used for outer-step NN training')
	return parser.parse_args()


def save_image(tensor, path, name):
	os.makedirs(path, exist_ok=True)
	image = TF.to_pil_image(tensor.cpu())
	image.save(os.path.join(path, name))

def add_noise(x_adv, x, l2):
	noise = torch.randn_like(x)
	current_l2_norm = torch.norm(noise, p=2)
	scaled_noise = (l2 / current_l2_norm) * noise
	x_adv = x_adv + scaled_noise.to(args.GPU_ID)
	return x_adv


class AdvDataset(torch.utils.data.Dataset):
	def __init__(self, input_dir=None, output_dir=None, targeted=False, eval=False):
		self.targeted = targeted
		self.data_dir = input_dir
		self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))
		self.filename = list(self.f2l.keys())

		if eval:
			self.data_dir = output_dir
			self.filename = []
			self.advfile = list(self.f2l.keys())
			for filename in self.advfile:
				filepath = os.path.join(self.data_dir, filename)
				if os.path.exists(filepath):
					self.filename.append(filename)
			print('=> Eval mode: evaluating on {}'.format(self.data_dir))
		else:
			self.data_dir = os.path.join(self.data_dir, 'images')
			print('=> Train mode: training on {}'.format(self.data_dir))
			print('Save images to {}'.format(output_dir))

	def __len__(self):
		return len(self.filename)

	def __getitem__(self, idx):
		filename = self.filename[idx]
		filepath = os.path.join(self.data_dir, filename)

		image = Image.open(filepath)
		image = image.resize((img_height, img_width)).convert('RGB')
		image = np.array(image).astype(np.float32) / 255
		image = torch.from_numpy(image).permute(2, 0, 1)
		label = self.f2l[filename]

		return image, label, filename

	def load_labels(self, file_name):
		dev = pd.read_csv(file_name)
		if self.targeted:
			f2l = {
				dev.iloc[i]['filename']: [dev.iloc[i]['label'], dev.iloc[i]['targeted_label']]
				for i in range(len(dev))
			}
		else:
			f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
		return f2l


def reshape_image(tensor, target_size=(224, 224)):
	target_h, target_w = target_size
	h, w = tensor.shape[-2], tensor.shape[-1]

	if h > target_h or w > target_w:
		crop_h = min(h, target_h)
		crop_w = min(w, target_w)
		top = max((h - crop_h) // 2, 0)
		left = max((w - crop_w) // 2, 0)
		tensor = tensor[:, :, top:top + crop_h, left:left + crop_w]

	pad_h = target_h - tensor.shape[-2]
	pad_w = target_w - tensor.shape[-1]
	if pad_h > 0 or pad_w > 0:
		tensor = FF.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

	return tensor


def wrap_model(model):
	if hasattr(model, 'default_cfg'):
		mean = model.default_cfg['mean']
		std = model.default_cfg['std']
	else:
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
	normalize = transforms.Normalize(mean, std)
	return torch.nn.Sequential(normalize, model)


def clamp(x, x_min, x_max):
	return torch.min(torch.max(x, x_min), x_max)


def to_jsonable(value):
	if isinstance(value, dict):
		return {key: to_jsonable(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [to_jsonable(item) for item in value]
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().tolist()
	return value


def load_pretrained_model(model_name, device):
	if model_name in models.__dict__.keys():
		print('=> Loading model {} from torchvision.models'.format(model_name))
		model = models.__dict__[model_name](weights='DEFAULT')
	else:
		try:
			import timm
		except Exception as exc:
			raise ValueError(
				'Model {} is not a torchvision model and timm could not be imported. '
				'Use a torchvision model or fix timm/torch installation.'.format(model_name)
			) from exc

		if model_name in timm.list_models():
			print('=> Loading model {} from timm.models'.format(model_name))
			model = timm.create_model(model_name, pretrained=True)
		else:
			raise ValueError('Model {} not supported'.format(model_name))

	return wrap_model(model.eval()).to(device)


def build_result_name(use_nn, use_ensemble, targeted):
	result_name = 'results_eval'
	if use_nn:
		result_name += '_nn'
	if use_ensemble:
		result_name += '_en'
	if targeted:
		result_name += '_tar'
	return result_name


class DisturbanceNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(6, 32, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		self.flow_head = nn.Conv2d(64, 2, 3, padding=1)
		self.res_head = nn.Conv2d(64, 3, 3, padding=1)
		self.param_head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(64, 4)
		)

	def forward(self, x_adv, theta):
		inp = torch.cat([x_adv, theta], dim=1)
		feat = self.encoder(inp)

		flow = self.flow_head(feat)
		noise = self.res_head(feat)
		flow = torch.tanh(flow) * 0.1

		bsz, _, hgt, wid = flow.shape
		grid_y, grid_x = torch.meshgrid(
			torch.linspace(-1, 1, hgt, device=flow.device),
			torch.linspace(-1, 1, wid, device=flow.device),
			indexing='ij'
		)
		base_grid = torch.stack((grid_x, grid_y), dim=-1)
		base_grid = base_grid.unsqueeze(0).repeat(bsz, 1, 1, 1)

		flow_grid = base_grid + flow.permute(0, 2, 3, 1)
		x_warp = F.grid_sample(x_adv, flow_grid, align_corners=True)

		x_dist = x_adv + 0.1 * (x_warp - x_adv) + 0.1 * noise

		params = self.param_head(feat)
		angle = torch.tanh(params[:, 0]) * 60
		scale = 1.0 + 1.0 * torch.tanh(params[:, 1])
		tx = torch.tanh(params[:, 2]) * 0.2
		ty = torch.tanh(params[:, 3]) * 0.2

		return {
			'dist': x_dist,
			'angle': angle,
			'scale': scale,
			'tx': tx,
			'ty': ty,
		}


class RAA(object):
	def __init__(self, model_name, targeted, n_theta=20, mu=0.4, step=100, device='cuda:0', ensemble=False, en_model_names=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, ensemble_resample_interval=10, use_nn=False, active_models_k=2, nn_inner_steps=5, nn_theta_bank=2):
		self.n_theta = n_theta
		self.mu = mu
		self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
		self.model = self.load_model(model_name).to(self.device)
		self.ensemble = ensemble
		self.use_nn = use_nn
		self.lambda1 = lambda1
		self.lambda2 = lambda2
		self.lambda3 = lambda3
		self.ensemble_resample_interval = max(1, int(ensemble_resample_interval))
		self.active_models_k = max(1, int(active_models_k))
		self.nn_inner_steps = max(1, int(nn_inner_steps))
		self.nn_theta_bank = max(1, int(nn_theta_bank))

		self.ensemble_models = []
		self.loss_models = [self.model]
		if self.ensemble:
			for en_model_name in en_model_names or []:
				model = self.load_model(en_model_name).to(self.device)
				self.ensemble_models.append(model)
				self.loss_models.append(model)

		self.active_models = [self.model]
		self.eps_delta = 16 / 255
		self.eps_theta = 16 / 255
		self.labda = 0.1
		self.target = targeted
		self.step = step
		self.lr = 1.6 / 255
		self.img_max, self.img_min = 1.0, 0.0

		self.h_net = DisturbanceNet().to(self.device)
		self.opt_h = torch.optim.Adam(self.h_net.parameters(), lr=1e-4)

	def load_model(self, model_name):
		return load_pretrained_model(model_name, self.device)

	def compute_loss(self, model, x, target):
		return FF.cross_entropy(model(x), target, reduction='mean')

	def update_active_models(self, outer_step):
		if not self.ensemble:
			self.active_models = [self.model]
			return self.active_models

		self.active_models = self.loss_models + [self.model]
		return self.active_models

	# def compute_hybrid_loss(self, x, target, active_models):
	# 	if len(active_models) == 1:
	# 		return self.compute_loss(active_models[0], x, target)

	# 	losses = [self.compute_loss(model, x, target) for model in active_models]
	# 	losses_stack = torch.stack(losses)

	# 	# stable normalization
	# 	mean = losses_stack.mean().detach()
	# 	losses_stack = losses_stack / (mean + 1e-8)

	# 	# soft balanced weights (KEY FIX)
	# 	temp = 0.7
	# 	weights = torch.softmax(losses_stack.detach() / temp, dim=0)

	# 	pareto_loss = (weights * losses_stack).sum()
	# 	sampled_loss = losses_stack.mean()

	# 	base_loss = self.lambda1 * sampled_loss + self.lambda3 * pareto_loss

	# 	# mild agreement encouragement
	# 	div = torch.std(losses_stack)

	# 	return base_loss - 0.05 * div

	def apply_disturbance_mapping(self, x_adv, theta):
		out = self.h_net(x_adv, theta)

		x_dist = out['dist']
		angle = out['angle']
		scale = out['scale']
		tx = out['tx']
		ty = out['ty']

		angle_rad = angle * (3.14159265 / 180.0)
		cos = torch.cos(angle_rad)
		sin = torch.sin(angle_rad)

		theta_affine = torch.zeros(x_dist.size(0), 2, 3, device=x_dist.device)
		theta_affine[:, 0, 0] = cos * scale
		theta_affine[:, 0, 1] = -sin * scale
		theta_affine[:, 1, 0] = sin * scale
		theta_affine[:, 1, 1] = cos * scale
		theta_affine[:, 0, 2] = tx
		theta_affine[:, 1, 2] = ty

		grid = F.affine_grid(theta_affine, x_dist.size(), align_corners=True)
		x_final = F.grid_sample(x_dist, grid, align_corners=True)
		x_final = torch.clamp(x_final, 0.0, 1.0)

		return x_final, x_dist, angle, scale, tx, ty

	def optimize_hnet_outer_step(self, x_adv, target, active_models):
		if not self.use_nn:
			return

		x_detached = x_adv.detach()
		theta_bank = []

		for _ in range(self.nn_theta_bank):
			theta = self.labda * torch.randn_like(x_detached)
			theta.requires_grad = True

			inp = x_detached + theta
			loss = FF.cross_entropy(self.model(inp), target, reduction='mean')
			grad = torch.autograd.grad(loss, theta)[0]

			theta = (theta + 0.1 * grad.sign()).detach()
			theta_bank.append(theta)

		for model in self.loss_models:
			for p in model.parameters():
				p.requires_grad = False

		self.h_net.train()
		for _ in range(self.nn_inner_steps):
			self.opt_h.zero_grad()
			total_loss = 0.0

			for theta in theta_bank:
				theta_input = theta.detach()
				x_input, x_dist, angle, scale, tx, ty = self.apply_disturbance_mapping(x_detached, theta_input)

				if self.ensemble:
					loss_h = torch.stack([
							self.compute_loss(model, x_input, target)
							for model in active_models
						]).mean()
				else:
					loss_h = self.compute_loss(self.model, x_input, target)

				reg = 0.01 * torch.mean(torch.abs(x_dist - x_detached))
				transform_loss = (
					torch.mean(torch.abs(angle))
					+ torch.mean(torch.abs(scale - 1.0))
					+ torch.mean(torch.abs(tx))
					+ torch.mean(torch.abs(ty))
				)

				if self.target:
					obj = loss_h + reg + 0.05 * transform_loss
				else:
					obj = -loss_h + reg + 0.05 * transform_loss

				# consistency regularization (NEW)
				consistency = torch.mean((x_input - x_detached) ** 2)

				total_loss = total_loss + obj + 0.05 * consistency

			total_loss = total_loss / len(theta_bank)
			total_loss.backward()
			self.opt_h.step()

		self.h_net.eval()
		for model in self.loss_models:
			for p in model.parameters():
				p.requires_grad = True

	def forward(self, x, target):
		x = x.to(self.device, non_blocking=True)
		if self.target:
			target = target[1].to(self.device, non_blocking=True)
		else:
			target = target.to(self.device, non_blocking=True)

		delta = torch.rand_like(x)
		delta = torch.clamp(delta, min=-self.eps_delta, max=self.eps_delta)

		step_bar = tqdm(total=self.step, desc='RAA steps', position=1, leave=False, dynamic_ncols=True)
		for i in range(self.step):
			x_adv = x + delta
			loss_print = 0.0
			self.model.zero_grad(set_to_none=True)
			sum_direction = torch.zeros_like(delta)

			active_models = self.update_active_models(i)
			if self.use_nn:
				self.optimize_hnet_outer_step(x_adv, target, active_models)

			theta_bar = tqdm(total=self.n_theta, desc='theta samples', position=2, leave=False, dynamic_ncols=True)
			for theta_idx in range(self.n_theta):
				theta = self.labda * torch.randn_like(x_adv)
				theta.requires_grad = True

				inp = x_adv + theta

				if self.ensemble:
					loss_theta = torch.stack([
						FF.cross_entropy(model(inp), target, reduction='mean')
						for model in active_models
					]).mean()
				else:
					loss_theta = FF.cross_entropy(self.model(inp), target, reduction='mean')

				grad_theta = torch.autograd.grad(loss_theta, theta)[0]

				grad_theta = grad_theta / (grad_theta.abs().mean() + 1e-8)
				grad_theta = grad_theta.detach()

				if self.target:
					thetanew = 0.05 * theta + self.mu * grad_theta
				else:
					thetanew = 0.05 * theta - self.mu * grad_theta

				if self.use_nn:
					with torch.no_grad():
						theta_input = thetanew.detach()
						h_theta, _, _, _, _, _ = self.apply_disturbance_mapping(x_adv.detach(), theta_input)
						# thetanew = (theta_input + h_theta).detach()
						delta_h = h_theta - x_adv.detach()
						delta_h = delta_h / (delta_h.abs().mean() + 1e-8)

						# thetanew = theta_input + 0.1 * delta_h
						if self.target:
							thetanew = theta_input + self.mu * grad_theta + 0.05 * delta_h
						else:
							thetanew = theta_input - self.mu * grad_theta + 0.05 * delta_h

				with torch.no_grad():
					if not self.ensemble:
						loss = self.compute_loss(self.model, x_adv + thetanew, target)
					else:
						losses = [
							FF.cross_entropy(model(x_adv + thetanew), target, reduction='mean')
							for model in active_models
						]
						losses = torch.stack(losses)

						# 🔥 smooth worst-case (BEST choice)
						temp = 5.0
						loss = torch.logsumexp(losses * temp, dim=0) / temp

					loss_view = loss.view(-1, 1, 1, 1)

					if self.target:
						direction_theta = loss_view * thetanew
					else:
						direction_theta = -loss_view * thetanew

					sum_direction += direction_theta

				loss_print += float(loss_theta.detach())
				theta_bar.update(1)
				if theta_idx % 5 == 0:
					theta_bar.set_postfix_str(f'loss={loss_print / max(1, theta_idx + 1):.4f}')

			theta_bar.close()

			if self.target:
				delta = delta - (sum_direction / self.n_theta).sign() * self.lr
			else:
				delta = delta + (sum_direction / self.n_theta).sign() * self.lr

			delta = torch.clamp(delta, min=-self.eps_delta, max=self.eps_delta)

			if i % 20 == 0:
				step_bar.set_postfix_str(f'loss={loss_print / max(1, self.n_theta):.4f}')
			step_bar.update(1)

			torch.cuda.empty_cache()

		step_bar.close()
		delta = clamp(delta, self.img_min - x, self.img_max - x)
		return delta, self.model


def main(args):
	os.makedirs(args.output_dir, exist_ok=True)
	device = torch.device(args.GPU_ID if torch.cuda.is_available() else 'cpu')

	dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=False)

	num_workers = min(4, os.cpu_count() or 0)
	loader_kwargs = {
		'batch_size': args.batchsize,
		'shuffle': False,
		'num_workers': num_workers,
		'pin_memory': torch.cuda.is_available(),
	}
	if num_workers > 0:
		loader_kwargs['persistent_workers'] = True
		loader_kwargs['prefetch_factor'] = 2

	dataloader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
	total_images = min(args.max_test_num, len(dataset))

	black_box_models = [model_name.strip() for model_name in args.bb_models.split(',') if model_name.strip()]
	transfer_models = {model_name: load_pretrained_model(model_name, device) for model_name in black_box_models}

	attacker = RAA(
		model_name=args.model,
		targeted=args.targeted,
		n_theta=args.n_theta,
		step=args.epoch,
		device=args.GPU_ID,
		ensemble=args.ensemble,
		en_model_names=[model_name.strip() for model_name in args.en_models.split(',') if model_name.strip()],
		lambda1=args.lambda1,
		lambda2=args.lambda2,
		lambda3=args.lambda3,
		ensemble_resample_interval=args.ensemble_resample_interval,
		use_nn=args.nn,
		active_models_k=args.active_models_k,
		nn_inner_steps=args.nn_inner_steps,
		nn_theta_bank=args.nn_theta_bank,
	)

	cnt = 0
	total = 0
	total_time = 0
	correct_lists = {'whitebox': []}
	pred_lists = {'whitebox': []}
	for model_name in black_box_models:
		correct_lists[model_name] = []
		pred_lists[model_name] = []
	result_rows = []
	transform_items = list(transform_dict.items())

	with tqdm(total=total_images, desc='Images processed', position=0, dynamic_ncols=True) as image_bar:
		for batch_idx, (images, labels, filenames) in enumerate(dataloader):
			if total >= args.max_test_num:
				break

			cnt_1 = 0
			start_time = time.time()

			perturbations, model = attacker.forward(images, labels)
			images = images.to(device, non_blocking=True)
			adv = images.detach() + perturbations.detach()
			labels_eval = labels[1] if args.targeted else labels
			labels_eval = labels_eval.to(device, non_blocking=True)

			with torch.inference_mode():
				transform_bar = tqdm(transform_items, desc=f'Transforms batch {batch_idx}', position=1, leave=False, dynamic_ncols=True)
				for transform_name, par_list in transform_bar:
					args.transform = transform_name
					transform_fn = getattr(Transform, transform_name)
					param_bar = tqdm(par_list, desc=f'{transform_name} params', position=2, leave=False, dynamic_ncols=True)
					for par in param_bar:
						args.transform_par = par

						if transform_name in ('resize_image', 'Random_Affine'):
							adv_test = reshape_image(transform_fn(adv, args.transform_par))
						else:
							adv_test = transform_fn(adv, args.transform_par)

						adv_test = adv_test.to(device, non_blocking=True)

						pred = model(adv_test)
						pred_classes_tensor = pred.argmax(dim=1)
						pred_classes = pred_classes_tensor.detach().cpu().tolist()
						if args.targeted:
							correct = int((labels_eval == pred_classes_tensor).sum().item())
						else:
							correct = int((labels_eval != pred_classes_tensor).sum().item())

						if batch_idx == 0:
							correct_lists['whitebox'].append(correct)
							pred_lists['whitebox'].append(list(pred_classes))
						else:
							correct_lists['whitebox'][cnt_1] += correct
							pred_lists['whitebox'][cnt_1].extend(list(pred_classes))

						for model_name, transfer_model in transfer_models.items():
							transfer_pred = transfer_model(adv_test)
							transfer_pred_classes_tensor = transfer_pred.argmax(dim=1)
							transfer_pred_classes = transfer_pred_classes_tensor.detach().cpu().tolist()
							if args.targeted:
								transfer_correct = int((labels_eval == transfer_pred_classes_tensor).sum().item())
							else:
								transfer_correct = int((labels_eval != transfer_pred_classes_tensor).sum().item())

							if batch_idx == 0:
								correct_lists[model_name].append(transfer_correct)
								pred_lists[model_name].append(list(transfer_pred_classes))
							else:
								correct_lists[model_name][cnt_1] += transfer_correct
								pred_lists[model_name][cnt_1].extend(list(transfer_pred_classes))

						cnt_1 += 1

					param_bar.close()
				transform_bar.close()

			img_indices = range(images.shape[0]) if args.save_all else range(1)
			for img_idx in img_indices:
				img_name_suffix = f'{batch_idx}_{img_idx}'
				save_image(adv[img_idx], f'{args.output_dir}/imgs', f'{args.attack}_new_{img_name_suffix}.png')

			total += labels_eval.shape[0]
			image_bar.update(labels_eval.shape[0])

			end_time = time.time()
			total_time += end_time - start_time

	asr_lists = {
		model_name: [x / total for x in values]
		for model_name, values in correct_lists.items()
	}

	for transform_name, par_list in transform_items:
		args.transform = transform_name
		for par in par_list:
			args.transform_par = par
			current_test = f'{args.attack}_{args.model}_{args.transform}'
			row_index = cnt
			result_rows.append({
				'Current Test': current_test,
				'params': to_jsonable(args.transform_par),
				'ASR_whitebox': to_jsonable(asr_lists['whitebox'][row_index]),
				**{f'ASR_{model_name}': to_jsonable(asr_lists[model_name][row_index]) for model_name in black_box_models},
				'time taken': to_jsonable(total_time),
				'Predicted List_whitebox': to_jsonable(pred_lists['whitebox'][row_index]),
				**{
					f'Predicted List_{model_name}': to_jsonable(pred_lists[model_name][row_index])
					for model_name in black_box_models
				},
			})
			cnt += 1

	result_name = build_result_name(args.nn, args.ensemble, args.targeted)
	results_txt_path = os.path.join(args.output_dir, f'{result_name}.txt')
	with open(results_txt_path, 'w', encoding='utf-8') as f:
		header = ['Current Test', 'params', 'ASR_whitebox']
		header.extend([f'ASR_{model_name}' for model_name in black_box_models])
		header.extend(['time taken'])
		f.write(', '.join(header) + '\n')
		for row in result_rows:
			values = [
				str(row['Current Test']),
				str(row['params']),
				str(row['ASR_whitebox']),
			]
			values.extend(str(row[f'ASR_{model_name}']) for model_name in black_box_models)
			values.append(str(row['time taken']))
			f.write(', '.join(values) + '\n')

	results_json_path = os.path.join(args.output_dir, 'results_eval.json')
	with open(results_json_path, 'w', encoding='utf-8') as f:
		json.dump(to_jsonable(result_rows), f, indent=2)


if __name__ == '__main__':
	args = get_parser()

	if torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True

	model_list = ['resnet50']
	attack_list = ['raa']
	transform_dict = {
		'adjust_brightness': [0.05, 0.1, 0.25, 0.5],
		'adjust_contrast': [0.05, 0.1, 0.25, 0.5],
		'JPEG_transform': [30, 50, 70, 90],
		'Gaussian_blur': [[5, 1.0]],
		'resize_image': [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2],
		'rotate_image': [0, 1, 2, 5, 10, 15, 20, 30, 45],
		'Random_perspective': [[0.25, 1.0], [0.5, 1.0]],
		'Random_Affine': [[10, [0.05, 0.05], [0.9, 0.9]]],
	}

	for model_name in model_list:
		args.model = model_name
		for attack in attack_list:
			args.attack = attack
			main(args)
