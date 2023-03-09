from deepsense_model import MyModel
import torch
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt



def finalTest(path, valFunc, val_loader, criterion, opt):
	d = torch.load(path)
	model = MyModel()
	model.load_state_dict(d['model'])

	sigma_vals = [0]
	#sigma_vals = [[0.05, 40,80], [0.1,50,80], [0.2,60,80]]
	accs = [] 
	for sigma in sigma_vals:
		model.sigma = sigma
		print(f'TESTING FOR sigma={sigma}')
		_, acc, _, _ = valFunc(val_loader, model, criterion, opt)
		accs.append(acc.item())
	plt.plot(sigma_vals, accs)
	plt.xlabel('sigma')
	plt.ylabel('accuracy')
	plt.show()


