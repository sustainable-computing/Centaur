from deepsense_model import MyModel
import torch
import matplotlib.pyplot as plt


def finalTest(path, valFunc, val_loader, criterion, opt):
	d = torch.load(path)
	model = MyModel().cuda()
	model.load_state_dict(d['model'])

	sigma_vals =[0.05, 0.1, 0.2, 0.3]
	#sigma_vals= [[0.05, 10,50], [0.1,20,50], [0.2,30,50]]
	#sigma_vals = [[10, 50], [20, 50], [30, 50]]
	accs = [] 
	for sigma in sigma_vals:
		model.sigma = sigma
		print(f'TESTING FOR sigma={sigma}')
		_, acc, _, _ = valFunc(val_loader, model, criterion, opt)
		accs.append(acc.item())
	plt.plot(sigma_vals, accs)
	plt.xlabel('sigma')
	plt.ylabel('accuracy')
	plt.savefig(f'd:\\plot{sigma}.jpg')
	plt.show()


