# Bayesian analysis of the empirical saturation point

We follow a Bayesian approach with conjugate distributions to extract the *empirical* saturation point from a range of Density Functional Theory (DFT) constraints, including those from relativistic mean field (RMF) theory and Skyrme energy density functionals. 

This repository supplements our manuscript in preparation.

We aim to provide statistically meaningful constraints on the nuclear saturation point to benchmark (and guide the construction of novel) microscopic interaction derived from chiral effective field theory (EFT).


## Installation

```shell
python3 -m venv env_satpoint
source env_satpoint/bin/activate
python3 -m pip install -r requirements_conj.txt  # for conjugate prior approach
python3 -m ipykernel install --name "satpoint"
jupyter-notebook&
# deactivate # when the job is done
```
You may need to specify the HDF5 directory, e.g., using the environment variable 
```shell
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.2_2  # the location may be different
```


## Overview

The repository is organized as follows:
* `data`: contains all DFT and EFT data files, including author and other relevant information
* `modules`: contains classes, functions, and more relevant to our analysis. It also provides a function to calculate and plot confidence regions of the bivariate t-distribution analytically. See the Appendices in our manuscript for more details.

The following Jupyter notebooks are included:
* `analysis_conjugate_priors.ipynb`: performs the conjugate prior analysis presented in the manuscript. The notebook supports parallel computing.
* `tutorial_conf_regions.ipynb`: provides a tutorial on plotting confidence regions of the t-distribution using the tools developed
* `saturation_analysis_mc.ipynb`: provides an independent implementation of our saturation analysis using brute-force Monte Carlo sampling. It can be used to check and generalize our analysis using conjugate priors. Another virtual environment with packages specified in `requirements_mc.txt` needs to be installed following the instructions above. This notebook was not used in our manuscript.
  

## Cite this work

```bibtex
@manual{saturationGitHub,
  author = {Christian Drischler},
  title = "{{Supplemental source code on GitHub}}",
  year = "2023",
  note = {\url{https://github.com/cdrischler/nuclear_saturation}},
  url = {\mbox{https://github.com/cdrischler/nuclear_saturation}}
}
```


## External (helpful) links

The following external resources may be helpful:

* [Murphy's notes](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf) on conjugate priors
* Lattimer's talk at the [INT 2022][LattimerINT:2022] (see slide 10)
* [Insights into nuclear saturation density from parity violating electron scattering][Horowitz:2020]
* [From finite nuclei to the nuclear liquid drop: Leptodermous expansion based on self-consistent mean-field theory][Reinhard:2005]
* [Nuclear charge and neutron radii and nuclear matter: trend analysis][Reinhard:2016]
* [Bayesian analysis][McDonnell:2015] of Skyrme functionals, which was updated [here][Schunck:2020].
* [Empirical saturation box][Drischler:2016] (see Section IV.B) used in microscopic calculations based on:
  * Table VII in [Dutra et al. (2012)][Dutra:2012]
  * Table I in [Brown & Schwenk][Brown:2013]
  * Table IV in [Kortelainen et al.][Kortelainen:2014]
* Analysis of RMF models by [Dutra et al. (2014)][Dutra:2014]


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
[Drischler2021:AnnRev]:https://www.annualreviews.org/doi/10.1146/annurev-nucl-102419-041903
