#!/bin/bash

cd pamap2
unzip PAMAP2_Datatset.zip
mv PAMAP2_Dataset/Protocol/* .
rm -rf PAMAP2_Dataset PAMAP2_Dataset.zip readme.pdf 
cd ..
