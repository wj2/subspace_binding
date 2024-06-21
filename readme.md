
### Dependencies 
This code relies on the usual python scientific computing environment and on
two of my other repositories. 
1. [general](https://github.com/wj2/general-neural): contains the decoding
   functions and other convenience functions
2. [composite_tangling](https://github.com/wj2/composite_tangling): contains 
   most of the functions for computing the theoretical quantities.

Once you have these two repositories on your path and all the additional python
packages downloaded, the code here should work. 

### Reproducing the analyses
The main two files for running the analyses are ```figures.py``` and 
```figures.conf``` -- the former contains all the code and the latter contains 
all the relevant parameters in a hierarchical configuration file. So, first 
change any relevant paths or parameters in ```figures.conf``` and then you can
run, for instance, most of the analyses in Figure 3 as follows:
```
import multiple_representations.figures as mrf

fig_key = 'general_theory'

gth_fig = mrf.GeneralTheoryFigure()
gth_fig.panel_k()
gth_fig.panel_n()
gth_fig.panel_snr(recompute=False)
gth_fig.panel_recovery()
fig_data[fig_key] = gth_fig.get_data()


gth_fig.save(fig_{}.svg'.format(fig_key))
``` 
where the ```panel``` methods typically construct a group of panels within the entire figure. This pattern holds for the rest of the figures.

### Generating figures from Johnston and Fine et al. (2023)
This code underlies all of the figures in Johnston and Fine et al. (2023) https://doi.org/10.48550/arXiv.2309.07766 . The data is separately available on figshare: https://doi.org/10.6084/m9.figshare.26065600 . The instructions above will generate all of the non-schematic and non-behavioral figures shown in the paper. Please feel free to contact [me](wjeffreyjohnston@gmail.com) if you have any questions. 
