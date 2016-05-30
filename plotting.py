# Python module to make plotting and fig exporting more convenient. 
# Lisa Neef / Helmholtz Centre for Ocean Research Kiel  

import matplotlib.pyplot as plt

def figexport(fig_name):

	"""
	This simple wrapper function takes whatever figure is currently open and exports 
	it as a pdf and a png file. 
	The files get put into a directory that's one "upstairs" from the current directory 
		and is called "Plots"
	It's good to have both because PDFs are vector graphics and good to use 
		in manuscripts, while PNGs are easier linked and shared. 

	INPUTS:
	fig_name: a string giving the filename of the figure, to which we append ".png" and ".pdf"  

	TODO:
	it would be really cool to add automatic upload to Imgur - makes it even easier to share plots 
	and embed them into markdown, etc. 
	"""

	fig_name_pdf = '../Plots/'+fig_name+'.pdf'
	fig_name_png = '../Plots/'+fig_name+'.png'
	
	print('saving figure as')
	print(fig_name_pdf)
	print(fig_name_png)

	plt.savefig(fig_name_pdf,dpi=96)
	plt.savefig(fig_name_png,dpi=96)
