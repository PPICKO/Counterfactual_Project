# Data Directory

Data files are not included in the repository due to size. Please download them manually.

## Adult Income Dataset (UCI)

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)

### Download Commands
```bash
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

### Direct Links
- Training set: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- Test set: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

## ACS Dataset (Folktables)

**Source:** [Folktables](https://github.com/socialfoundations/folktables)

### Option 1: Using Folktables (Recommended)
```python
pip install folktables

from folktables import ACSDataSource

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=['CA'], download=True)
```

### Option 2: Manual Download
1. Go to [Census Bureau PUMS](https://www.census.gov/programs-surveys/acs/microdata.html)
2. Navigate to: PUMS > 2018 > 1-Year > California
3. Download `psam_p06.csv`
4. Place in `data/2018/1-Year/psam_p06.csv`

## Expected Structure
```
data/
├── adult.data
├── adult.test
└── 2018/
    └── 1-Year/
        └── psam_p06.csv
```
