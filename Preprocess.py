import numpy as np
import pandas as pd
import re
import argparse

#Comandline Aruguent
parser=argparse.ArgumentParser(description='Preprocess for XML-ridge')
parser.add_argument('-data',type=str,default=None)
parser.add_argument('-feature',type=str,default='BoW',help='select BoW or TF-IDF')

args=parser.parse_args()


class DataProcess:
    def __init__(self,file_path):
        self.file_path=file_path
        self.df=None
    def point_label(self):
        with open(self.file_path,'r') as file:
            lines=file.readlines()
        lines.pop(0)#eliminate the first line
        label_list=[]#for each point's labels
        for line in lines:
            label_feature=line.split(' ')
            labels=label_feature[0]
            labels=labels.split(',')
            label_list.append(labels)
        self.df=pd.DataFrame([(i+1, label) for i, row in enumerate(label_list) for label in row], columns=["point", "label"])
    def point_feature(self):
        with open(self.file_path,'r') as file:
            lines=file.readlines()
            lines.pop(0)
        feature_list=[]
        for line in lines:
            features=line.split(' ')
            features.pop(0)
            point_list=[]
            for feature in features:
                feature=re.sub(r':[^ ]*', '',feature)
                point_list.append(feature)
            point_list=','.join(point_list)
            cleaned_list=re.sub(r'\n', '', point_list)
            point_list=cleaned_list.split(',')
            feature_list.append(point_list)
        df_feature = pd.DataFrame([(i+1, label) for i, row in enumerate(feature_list) for label in row], columns=["point", "feature"])
        return df_feature
    def TF_IDF(self):
        with open(self.file_path,'r') as file:
            lines=file.readlines()
            lines.pop()
        TF_IDF_list=[]
        for line in lines:
            features=line.split(' ')
            features.pop(0)
            TF_IDF=[]
            for feature in features:
                num = re.sub(r'.*:', '', feature)
                num=re.sub(r':','',num)
                TF_IDF.append(num)
            features = ','.join()



data=args.data
#data_dir='/home/hayashi/categorize/data/'+data
data_dir='./data_dir/'+data
if args.data=='Bibtex':
    train_processor=DateProcessor(dir+'/train.txt')
    train_processor.fit()
elif args.data=='Delicous200K':
    train_processor=DateProcessor(dir+'/train.txt')