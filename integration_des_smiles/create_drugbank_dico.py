# libraries 
import os
import gzip
import pandas as pd
from unidecode import unidecode
import xml.etree.ElementTree as ET

#set path as the path in which this file is located
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# new drugbank database
drugbank = gzip.open('../data/drugbank2024.xml.gz','rt',encoding="utf-8")
tree = ET.parse(drugbank)
root = tree.getroot()

# create our new drugbank database
# drugbank_dico = gzip.open('../data/drugbank_dico.txt.gz', 'wt')
drugbank_dico = open('../data/drugbank_dico.txt', 'wt')
for drug in root.iterfind('{http://www.drugbank.ca}drug'):
    
    # check if there is a SMILES, because if not we don't put it in our database
    properties = drug.find('{http://www.drugbank.ca}calculated-properties')

    if properties != None:
        for property in properties.iterfind('{http://www.drugbank.ca}property'):
            if property.find('{http://www.drugbank.ca}kind').text == 'SMILES':
                smiles = property.find('{http://www.drugbank.ca}value').text
                if smiles == None:
                    continue # if the SMILES is None we go to the next drug
    else:
        continue # if there are no properties we go to the next drug

    # drug id
    id = drug.find('{http://www.drugbank.ca}drugbank-id').text

    names_synonyms_brands_products = []

    # drug name
    name = drug.find('{http://www.drugbank.ca}name').text.upper()
    names_synonyms_brands_products.append(name)

    # drug synonyms
    synonyms = drug.find('{http://www.drugbank.ca}synonyms')
    for synonym in synonyms.iterfind('{http://www.drugbank.ca}synonym'):
        if synonym.text.upper() not in names_synonyms_brands_products:
            names_synonyms_brands_products.append(synonym.text.upper())

    # drug international brand name
    international_brands = drug.find('{http://www.drugbank.ca}international-brands')
    for international_brand in international_brands.iterfind('{http://www.drugbank.ca}international-brand'):
        brand = international_brand.find('{http://www.drugbank.ca}name').text.upper()
        if brand not in names_synonyms_brands_products:
            names_synonyms_brands_products.append(brand)

    products = drug.find('{http://www.drugbank.ca}products')
    for product in products.iterfind('{http://www.drugbank.ca}product'):
        product_name = product.find('{http://www.drugbank.ca}name').text.upper()
        if product_name not in names_synonyms_brands_products:
            names_synonyms_brands_products.append(product_name)

    # write the string to the "drugbank_dico" output file
    drugbank_dico.write(f'{id}\t{smiles}\t{names_synonyms_brands_products}\n')
