RiskProp: Collision-Anchored Self-supervised Temporal Constraints for Early Accident Anticipation

### Installation

1. Create a new conda environment:

```bash
conda create -n mmaction2 python=3.12
conda activate mmaction2
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

1. Train model

first change the 'name' and 'CUDA_VISIBLE_DEVICES',then

```bash
./dist_train.sh
```

2. Test model

```bash
./dist_test.sh
```