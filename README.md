# advSyncOA
This repository contains code for the research paper "Analytical foundation for adversarial synchronization control in oscillator networks."

## Terms of Use
This project is licensed under the MIT License. When using this code, please cite our paper:

Takemoto K (2026) **Analytical Foundation for Adversarial Synchronization Control in Oscillator Networks.** arXiv:2605.14492. doi: [10.48550/arXiv.2605.14492](https://doi.org/10.48550/arXiv.2605.14492).

## Requirements
- Python 3.11

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Synchronization transition (all-to-all, ER, and BA networks)

#### Run simulations
```bash
bash run_oa_verification.sh --kave 12 --N_net 1000 --N_alltoall 1000
```

To skip the all-to-all case and run only network simulations:
```bash
bash run_oa_verification.sh --skip_alltoall --kave 12 --N_net 1000
```

#### Plot results (OA theory vs simulations)
```bash
python analyze_oa_verification.py --results_dir results_oa_N1000_kave12 --N_net 1000 --kave 12
```

### Fixed-point structure
```bash
python fig_fixedpoint.py
```


### Kramers escape verification
 
#### Run simulations
```bash
bash run_kramers.sh
```
 
#### Compute escape probability and plot
```bash
python check_kramers_slope.py
```


### Degree-resolved order parameter (BA network)
```bash
python gen_Rk_BA.py
```