#日経平均株価をPythonのライブラリから１０年分自動取得して、予測するプログラム
import streamlit as st
import time
t1 = time.time()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

# from __future__ import print_function
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import plot_model

from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import orthogonal
from keras.initializers import TruncatedNormal
from pandas_datareader import data as web
import math
import pandas as pd

st.title('日経平均予測アプリ(LSTM)')
st.markdown('## 概要及び注意事項')
st.write("当アプリでは、翌営業日の日経平均の終値が前日終値よりも上昇するか、下落するかを過去データによりLSTMを使用して予測します。ただし本結果により投資にいかなる損失が生じても、当アプリでは責任を取りません。あくまで参考程度にご利用ください。")

if st.button('予測開始'):
    try:
        comment = st.empty()
        comment.write('予測を開始します。最大20分程度お待ちください。')
        #データセットを準備する
        n225 = web.DataReader("NIKKEI225", "fred")
        n225 = n225.dropna()

        lastday = n225[-1:]
        lastday = lastday.index.tolist()
        lastday = map(str, lastday)
        lastday = ''.join(lastday)
        lastday = lastday.rstrip("00:00:00")

        date = n225[1:]
        date = date.index.tolist()
        len(date)

        #データフレーム型からリスト型に変換して、価格のみを取り出す
        n225 = n225.values.tolist()
        #2Dを１Dに変換する
        n225arr = np.array(n225)
        n225arr = n225arr.ravel()
        n225 = n225arr.tolist()

        ln_n225 = []
        for line in n225:
            ln_n225.append(math.log(line))
        count_s = len(ln_n225)    

        # 株価の上昇率を算出、おおよそ-1.0-1.0の範囲に収まるように調整
        modified_data = []
        for i in range(1, count_s):
            modified_data.append(float(ln_n225[i] - ln_n225[i-1])*100)
            
        #ラベルづけ
        count_m = len(modified_data)
        answers = []
        for j in range(count_m):
            if modified_data[j] > 0:
                answers.append(1)
            else:
                answers.append(0)

        x_dataset = pd.DataFrame()
        x_dataset['modified_data'] = modified_data
        x_dataset.to_csv('x-data.csv', index = False)

        t_dataset = pd.DataFrame()
        t_dataset['answers'] = answers
        t_dataset.to_csv('t-data.csv', index = False)

        # 学習データ
        df1 = csv.reader(open('x-data.csv', 'r'))
        data1 = [ v for v in df1]
        mat = np.array(data1)
        mat2 = mat[1:]  # 見出し行を外す
        x_data = mat2.astype(np.float)  # 2float変換

        # ラベルデータ
        # 1％以上／0％以上／-1％以上／-1％未満
        df2 = csv.reader(open('t-data.csv', 'r'))
        data2 = [ w for w in df2]
        mat3 = np.array(data2)
        mat4 = mat3[1:]                      # 見出し行を外す
        t_data = mat4.astype(np.int)  # float変換

        maxlen = 80              # 入力系列数
        n_in = x_data.shape[1]   # 学習データ（＝入力）の列数
        n_out = t_data.shape[1]  # ラベルデータ（=出力）の列数
        len_seq = x_data.shape[0] - maxlen + 1
        data = []
        target = []
        for i in range(0, len_seq):
          data.append(x_data[i:i+maxlen, :])
          target.append(t_data[i+maxlen-1, :])

        x = np.array(data).reshape(len(data), maxlen, n_in)
        t = np.array(target).reshape(len(data), n_out)

        # ここからソースコードの後半
        n_train = int(len(data)*0.9)              # 訓練データ長
        x_train,x_test = np.vsplit(x, [n_train])  # 学習データを訓練用とテスト用に分割
        t_train,t_test = np.vsplit(t, [n_train])  # ラベルデータを訓練用とテスト用に分割

        class Prediction :
          def __init__(self, maxlen, n_hidden, n_in, n_out):
            self.maxlen = maxlen
            self.n_hidden = n_hidden
            self.n_in = n_in
            self.n_out = n_out

          def create_model(self):
            model = Sequential()
            model.add(LSTM(self.n_hidden, batch_input_shape = (None, self.maxlen, self.n_in),
                    kernel_initializer = glorot_uniform(seed=20170719), 
                    recurrent_initializer = orthogonal(gain=1.0, seed=20170719), 
                    dropout = 0.5, 
                    recurrent_dropout = 0.5))
            model.add(Dropout(0.5))
            model.add(Dense(self.n_out, 
                    kernel_initializer = glorot_uniform(seed=20170719)))
            model.add(Activation("sigmoid"))
            model.compile(loss="binary_crossentropy", optimizer = "Adam", metrics = ['accuracy'])
            return model

          # 学習
          def train(self, x_train, t_train, batch_size, epochs) :
            model = self.create_model()
            model.fit(x_train, t_train, batch_size = batch_size, epochs = epochs, verbose = 1,
                  shuffle = True, validation_split = 0.1)
            return model

        n_hidden = 80     # 出力次元
        epochs = 100   # エポック数
        batch_size = 10   # ミニバッチサイズ

        # モデル定義
        prediction = Prediction(maxlen, n_hidden, n_in, n_out)
        # 学習
        model = prediction.train(x_train, t_train, batch_size, epochs)
        # テスト
        score = model.evaluate(x_test, t_test, batch_size = batch_size, verbose = 1)

        # 正答率、準正答率（騰落）集計
        preds = model.predict(x_test)
        correct = 0
        semi_correct = 0
        for i in range(len(preds)):
          pred = preds[i] 
          tar = t_test[i]
          if int(pred) == tar :
            correct += 1

        predicted = int(preds[-1:])

        Positive_Solution_Rate = round(correct / len(preds)*100, 2)
        lastday = str(lastday)
      
        if predicted == 1:
            result = '「上昇」するでしょう'
        else:
            result = '「下落」するでしょう'
        
        t2 = time.time()
        elapsed_time = t2- t1
        elapsed_time = round(elapsed_time, 2)
        seconds = elapsed_time
        minutes = int(seconds/60)
        seconds = seconds % 60
        hours = int(minutes/60)
        minutes = minutes % 60
        hours = str(hours)
        minutes = str(minutes)
        seconds = str(round(seconds))

        st.write(f'{lastday}の翌営業日の日経平均は、')
        st.write(result)
        st.write(f'正解率:{Positive_Solution_Rate}%')
        st.write("プログラム処理時間")
        st.write(f'{hours}時間{minutes}分{seconds}秒')
        comment.write('完了しました！')
    except:
        st.error('エラーが発生しました。申し訳ありません。もうしばらくしてから、再度実行してください。')
