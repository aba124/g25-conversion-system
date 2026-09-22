Converts consumer DNA raw-data file into 25 ancestry coordinates as per Global25, both scaled and unscaled, using a PCA model built from public reference genotypes. 

Model is built on your own machine from [Allen Ancient DNA Resource](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FFIDCW) and everything afterwards runs offline on your own computer. 

All you need is Python 3.9+ and numpy.

requirements.txt
run.txt

## Install

```bash
git clone https://github.com/aba124/g25-conversion-system.git
cd g25-conversion-system
py -m pip install -r requirements.txt
py selftest.py
```

## Setup

2 steps are not in the repo

```bash
py fetch_reference.py --out data/aadr
py build_model.py --panel data/aadr/v66_HO --out models/ho25
```



## Use

```bash
py convert.py yourfile.txt --model models/ho25 --name Sample
```

Reads AncestryDNA, 23andMe, FamilyTreeDNA, MyHeritage, LivingDNA and VCF, prints coordinates as scaled and unscaled

Use the scaled coordinates for distances and admixture models

Check the accuracy yourself:

```bash
py validate.py --model models/ho25 --chip yourfile.txt
```

## Privacy

Raw data and coordinates are personal info. Ignore excludes vendor filenames, g25, .txt data/ and models/, however look at status before commit and ensure no file is already staged

## Limitations

Not Davidski's, never published PCA and not reproducable, accuracy depends on how many SNPs you share with the panel.