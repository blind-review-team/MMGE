from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb

class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data', self.dataset_name)
        # 打印路径确认
        print(f"数据集路径: {self.dataset_path}, 存在: {os.path.exists(self.dataset_path)}")

        # dataframe
        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']
        
        # 添加评分字段
        self.use_rating = config['use_rating'] if 'use_rating' in config else False
        if self.use_rating:
            self.rating_field = self.config['RATING_FIELD']
            self.min_rating = self.config['min_rating']
            self.max_rating = self.config['max_rating']
            self.normalize_ratings = self.config['normalize_ratings']
        
        # 物品流行度存储
        self.item_popularity = None

        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1
        
        # 如果使用评分，计算物品流行度
        if self.use_rating:
            self.calculate_item_popularity()
            if self.normalize_ratings:
                self.normalize_rating_values()

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label, 'timestamp']
        
        # 如果使用评分，添加评分字段到需要加载的列
        if self.use_rating and self.rating_field:
            cols.append(self.rating_field)
            
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def calculate_item_popularity(self):
        """计算物品流行度 = 物品出现次数/总交互次数"""
        item_counts = self.df[self.iid_field].value_counts()
        total_interactions = len(self.df)
        self.item_popularity = item_counts / total_interactions
        # 将结果转为字典形式，方便查询
        self.item_popularity = self.item_popularity.to_dict()
        
    def normalize_rating_values(self):
        """将评分数据标准化到[0,1]范围"""
        if self.rating_field in self.df.columns:
            self.df[self.rating_field+'_normalized'] = (self.df[self.rating_field] - self.min_rating) / (self.max_rating - self.min_rating)
        
    def get_item_popularity(self, item_id):
        """获取物品流行度"""
        if self.item_popularity is None:
            return 0.0
        return float(self.item_popularity.get(item_id, 0))
        #     return 0
        # return self.item_popularity.get(item_id, 0)
    
    def get_normalized_rating(self, user_id, item_id):
        """获取用户对物品的标准化评分"""
        if not self.use_rating or self.rating_field+'_normalized' not in self.df.columns:
            return 0.5  # 返回中间值作为默认值

        try:
            # 查找用户-物品评分
            rating_row = self.df[(self.df[self.uid_field] == user_id) & (self.df[self.iid_field] == item_id)]
            if len(rating_row) > 0:
                rating = rating_row[self.rating_field + '_normalized'].values[0]
                # 确保返回浮点数
                return float(rating)
            return 0.5  # 如果没有找到评分，返回中间值
        except Exception as e:
            print(f"获取评分时出错: {e}, 用户ID: {user_id}, 物品ID: {item_id}")
            return 0.5  # 发生错误时返回中间值
        #     return 0
        #
        # rating_row = self.df[(self.df[self.uid_field] == user_id) & (self.df[self.iid_field] == item_id)]
        # if len(rating_row) > 0:
        #     return rating_row[self.rating_field+'_normalized'].values[0]
        # return 0
        
    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        
        # 复制评分处理相关的属性
        if self.use_rating:
            nxt.item_popularity = self.item_popularity
            
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
