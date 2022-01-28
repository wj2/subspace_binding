
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
run, for instance, all the decoding analyses as follows:
```
import multiple_representations.figures as mrf

dec_fig = mrf.DecodingFigure(data=fig_data.get(fig_key))
dec_fig.panel_rwd_generalization()
dec_fig.panel_prob_generalization()
``` 
where the first method call does the reward decoding analyses and the second 
does the probability decoding analyses. To fit coefficients, using the same 
object, you run:
```
dec_fig.panel_rwd_model_prediction(force_recompute=True,
                                   force_refit=True)
dec_fig.panel_prob_model_prediction(force_recompute=True,
                                    force_refit=True)
```
and so on. 

For the theoretical quantities on coefficients you have fit already, you can
use the ```TheoryFigure``` and its associated methods. 
