# Code for paper: "Algorithm Selection on a Meta Level"

This repository holds the code for our paper "Algorithm Selection on a Meta Level" by Alexander Tornede, Lukas Gehring, Tanja Tornede, Marcel Wever and Eyke Hüllermeier. Regarding questions please contact alexander.tornede@upb.de .

Please cite this work as
```
@inproceedings{tornede2021as_on_meta_level,
  title={Algorithm Selection on a Meta Level},
  author={Tornede, Alexander and Gehring, Lukas and Tornede, Tanja and Wever, Marcel and H{\"u}llermeier, Eyke},
  booktitle={Machine Learning},
  year={2022}
}
```

## Abstract
The problem of selecting an algorithm that appears most suitable for a specific instance of an algorithmic problem class, such as the Boolean satisfiability problem, is called instance-specific algorithm selection. Over the past decade, the problem has received considerable attention, resulting in a number of different methods for algorithm selection. Although most of these methods are based on machine learning, surprisingly little work has been done on meta learning, that is, on taking advantage of the complementarity of existing algorithm selection methods in order to combine them into a single superior algorithm selector. In this paper, we introduce the problem of meta algorithm selection, which essentially asks for the best way to combine a given set of algorithm selectors. We present a general methodological framework for meta algorithm selection as well as several concrete learning methods as instantiations of this framework, essentially combining ideas of meta learning and ensemble learning. In an extensive experimental evaluation, we demonstrate that ensembles of algorithm selectors can significantly outperform single algorithm selectors and have the potential to form the new state of the art in algorithm selection.

## Execution Details (Getting the Code to Run)
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper. For reproducing the meta learning results (Section 4), please refer to [this file](meta_learning/README.md). For reproducing the ensemble learning results (Section 5), please refer to [this file](ensemble_learning/README.md).
