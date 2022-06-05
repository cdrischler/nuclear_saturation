# Nuclear saturation point

We use Bayesian model mixing to quantify the uncertainties in the empirical nuclear saturation point of symmetric matter. We also quantity how well recent microscopic calculations reproduce the empirical point.

## Installation

```shell
python3 -m venv env_satpoint
source env_satpoint/bin/activate
pip3 install -r requirements.txt
ipython kernel install --name "satpoint" --user
jupyter-notebook&
# deactivate # when the job is done
```

## References

### Relevant for the data sets we use
* [Bayesian analysis][McDonnell:2015] of Skyrme functionals, which was updated [here][Schunck:2020].
* [Empirical saturation box][Drischler:2016] (see Section IV.B) used in microscopic calculations based on:
  * Table VII in [Dutra et al. (2012)][Dutra:2012]
  * Table I in [Brown & Schwenk][Brown:2013]
  * Table IV in [Kortelainen et al.][Kortelainen:2014]
* Analysis of RMF models by [Dutra et al. (2014)][Dutra:2014] could be used here as well.

### Additional
* Lattimer's talk at the [INT 2022][LattimerINT:2022] (see slide 10)
* [Insights into nuclear saturation density from parity violating electron scattering][Horowitz:2020]
* [From finite nuclei to the nuclear liquid drop: Leptodermous expansion based on self-consistent mean-field theory][Reinhard:2005]
* [Nuclear charge and neutron radii and nuclear matter: trend analysis][Reinhard:2016]

### Kernel Density Estimation (KDE)
* https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/
* https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
* https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67
* https://www.python-graph-gallery.com/2d-density-plot/
* https://seaborn.pydata.org/generated/seaborn.kdeplot.html
* https://stackoverflow.com/questions/4150171/how-to-create-a-density-plot-in-matplotlib
* https://www.delftstack.com/howto/matplotlib/matplotlib-density-plot/

[McDonnell:2015]:https://arxiv.org/abs/1501.03572
[Schunck:2020]:https://arxiv.org/abs/2003.12207
[Drischler:2016]:https://arxiv.org/abs/1510.06728
[Brown:2013]:https://arxiv.org/abs/1311.3957
[Dutra:2014]:https://arxiv.org/abs/1405.3633
[Dutra:2012]:https://arxiv.org/abs/1202.3902
[Kortelainen:2014]:https://arxiv.org/abs/1312.1746
[LattimerINT:2022]:https://www.int.washington.edu/sites/default/files/schedule_session_files/Lattimer%2C%20J.pdf

[Horowitz:2020]:https://arxiv.org/abs/2007.07117
[Reinhard:2005]:https://arxiv.org/abs/nucl-th/0510039
[Reinhard:2016]:https://arxiv.org/abs/1601.06324
