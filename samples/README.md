Put raw DNA files here to test them

Everything in this folder is gitignored except this file so nothing personal
can be committed by accident

Convert one

    py convert.py samples/whatever.txt --model models/ho25 --name Whatever

Convert every file in here at once and build a combined datasheet

    py batch.py --dir samples --model models/ho25

Accepted formats are AncestryDNA 23andMe FamilyTreeDNA MyHeritage LivingDNA
tellmeGen plain rsid csv or tsv and single sample VCF including gzipped
