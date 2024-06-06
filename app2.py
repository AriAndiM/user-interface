# Terminal Alt+1
import streamlit as st
import streamlit.components.v1 as html
import pandas as pd
import numpy as np
import string
import re
import pickle
import joblib
import warnings
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from collections import defaultdict
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
# string.punctuation
nltk.download('punkt')
nltk.download('stopwords')

with st.container():
    st.markdown('<h1 style = "text-align: center;"> Pengaruh Penggunaan Kata Sifat Pada Analisis Sentimen Ulasan Pantai Jawa Timur Menggunakan Skip-gram dan Support Vector Machine (SVM) </h1>', unsafe_allow_html = True)
    teks_input = st.text_area("Masukkan Teks")
    submit = st.button("Submit")
    if submit:
        if teks_input:
            df_mentah = pd.DataFrame({'Ulasan': [teks_input]})
            # df_mentah

            # Cleaning
            def cleaning(text):
              # untuk mengganti link menjadi spasi
              ulasan = re.sub(r'https\S*', ' ', text)
              # untuk mengganti tanda baca dengan spasi
              ulasan = re.sub(r'[{}]'.format(string.punctuation), ' ', ulasan)
              # untuk mengganti karakter selain a-z seperti emot menjadi spasi
              ulasan = re.sub('[^a-zA-Z]', ' ', ulasan)
              # mengganti newline dengan spasi
              ulasan = re.sub('\n', ' ', ulasan)
              # mengganti kata hanya 1 huruf dengan spasi
              ulasan = re.sub(r'\b[a-zA-Z]\b', ' ', ulasan)
              # menghilangkan spasi berlebih
              ulasan = ' '.join(ulasan.split())
              return ulasan

            def case_folding(text):
              ulasan = text.lower()
              return ulasan

             # Tokenisasi
            def tokenization(text):
                token = word_tokenize(text)
                return ','.join(token)

            # Normalisasi
            path_kbba = 'Kamus/kbba.xlsx'
            kbba = pd.read_excel(path_kbba)

            kbba_dict = {}
            for index,row in kbba.iterrows():
              kbba_dict[row['slang']] = row['baku']

            def normalization(text):
              normalized_token = []
              for kata in text.split(','):
                # print(text)
                if kata in kbba_dict:
                  baku = kbba_dict[kata]
                  # print(text, '->', baku)
                  banyak_kata = len(baku.split())
                  if banyak_kata > 1:
                    kata_baku = baku.split()
                    for idx, row in enumerate(kata_baku):
                      normalized_token.append(kata_baku[idx])
                  else:
                    normalized_token.append(baku)

                if kata not in kbba_dict:
                  normalized_token.append(kata)

              return ','.join(normalized_token)

            # Stopword removal
            stoplist = stopwords.words('indonesian')

            def stopword_removal(x):
              hasil = []
              list_kata = ["lebih", "kurang", "paling", "tidak", "sangat", "banget","terlalu","amat"]
              # memisahkan antar kata berdasarkan spasi
              for i in x.split(','):
                # dilakukan pengecekan pada stoplist
                if i not in stoplist or i in list_kata:
                  # memasukkan kata yang tidak ada di stoplist ke array
                  hasil.append(i)
              return ','.join(hasil)

            # Stemming
            Fact = StemmerFactory()
            Stemmer = Fact.create_stemmer()

            def stemming(text):
              hasil = []
              for kata in text.split(','):
                stemming_kata = Stemmer.stem(kata)
                hasil.append(stemming_kata)
              return ','.join(hasil)

            df_mentah['Clean']= df_mentah['Ulasan'].apply(cleaning)
            df_clean = df_mentah[['Clean']]
            df_clean['Case Folding']= df_clean['Clean'].apply(case_folding)
            # st.write(df_clean['Case Folding'])
            df_clean['Tokenization'] = df_clean['Case Folding'].apply(tokenization)
            # st.write(df_clean['Tokenization'])
            df_clean['Normalization'] = df_clean['Tokenization'].apply(normalization)
            # st.write(df_clean['Normalization'])
            df_clean['Stopword Removal']=df_clean['Normalization'].apply(stopword_removal)
            # st.write(df_clean['Stopword Removal'])
            df_clean['Stemming'] = df_clean['Stopword Removal'].apply(stemming)
            # st.write(df_clean['Stemming'])

            # pembentukan unigram bigram kata sifat
            def cek_kata_sifat(kata):
              file_path = 'Kamus/Kata Sifat Full.csv'
              df_kata_sifat = pd.read_csv(file_path)
              # Mengecek apakah kata ada dalam kolom 'Kata' dari dataframe
              kata_ditemukan = kata in df_kata_sifat['Kata'].values

              if kata_ditemukan:
                return True
              else:
                return False

            # Kata2-kata2 yang harus muncul sebelum kata sifat
            before_adj = ["lebih", "kurang", "paling", "tidak", "sangat", "amat", "terlalu"]

            # Kata-kata yang harus muncul setelah kata sifat
            after_adj = ["banget"]

            full = ["lebih", "kurang", "paling", "tidak", "sangat", "amat", "terlalu", "banget"]

            uni_bi = []
            not_uni = defaultdict(set)  # Menggunakan defaultdict untuk menyimpan set berdasarkan indeks

            for index, row in df_clean.iterrows():
                ulasan = row['Stemming']
                words = ulasan.split(',')
                # print(words)
                ind_uni_bi = []
                for idx, kata in enumerate(words):
                    if kata in before_adj:
                        # Menampilkan kata2 setelahnya jika ada
                        if idx + 1 < len(words):
                            kata_setelah = words[idx + 1]
                            if kata_setelah not in full and cek_kata_sifat(kata_setelah):
                                bigram = f"{kata} {kata_setelah}"
                                # print(f"Bigram Before: {bigram}")
                                not_uni[index].add(kata_setelah)
                                ind_uni_bi.append(bigram)
                            else:
            #                     ind_uni_bi.append(kata_setelah)
                                ind_uni_bi.append(kata)

                    elif kata in after_adj:
                        # Menampilkan kata2 sebelumnya jika ada
                        if idx - 1 >= 0:
                            kata_sebelum = words[idx - 1]
                            if kata_sebelum not in full and cek_kata_sifat(kata_sebelum):
                                bigram = f"{kata_sebelum} {kata}"
                                # print(f"Bigram After: {bigram}")
                                not_uni[index].add(kata_sebelum)
                                ind_uni_bi.append(bigram)
                            else:
            #                     ind_uni_bi.append(kata_sebelum)
                                ind_uni_bi.append(kata)

                    elif kata in not_uni[index] or kata not in not_uni[index]:
                        # print(f"Unigram: {kata2}")
                        ind_uni_bi.append(kata)

                # Menyimpan indeks kata2 pada uni_bi2
                uni_bi.append(ind_uni_bi)

            idx_dok = []
            idx_kata = []
            kata_sifat_bigram = []

            # Menyimpan indeks kata dalam set untuk setiap dokumen
            for idx, words_set in not_uni.items():
                ind_dok_int = []
                ind_kata_int = []
                kata_int_list = []
                for kata in words_set:
                    # Mencari indeks kata dalam setiap dokumen
                    for i, kata_dalam_dokumen in enumerate(uni_bi[idx]):
                        if kata_dalam_dokumen == kata:
                            ind_dok_int.append(idx)
                            ind_kata_int.append(i)
                            kata_int_list.append(kata_dalam_dokumen)  # Append to the list
                            break  # Exit the loop once the kata is found
                idx_dok.append(ind_dok_int)
                idx_kata.append(ind_kata_int)
                kata_sifat_bigram.append(kata_int_list)  # Append the list to store all kata for the current iteration

            def menghapus_kata(uni_bi):
              for i, kata_dalam_dokumen in enumerate(uni_bi):
                  for j, kata in enumerate(kata_dalam_dokumen):
                      kata_ada = kata in kata_sifat_bigram[i]
                      # hasil_cek.append(kata_ada)
                      if kata_ada:
                          if j + 1 < len(uni_bi[i]) and (len(uni_bi[i][j+1].split())) == 2 or (len(uni_bi[i][j-1].split())) == 2:
                            # print('idx :',i,'-',j,'=',uni_bi[i][j])
                            del uni_bi[i][j]

            menghapus_kata(uni_bi)

            unigram_bigram = [{'Unigram Bigram': ulasan} for ulasan in uni_bi]
            unigram_bigram = pd.DataFrame(unigram_bigram)
            # st.write('Hasil Pembentukan unigram bigram\n',unigram_bigram)

            # skip-gram
            model_w2v = Word2Vec.load("Ekstraksi Fitur/skenario 2 model w7.model")

            data = unigram_bigram['Unigram Bigram']
            hasil_preprocess = []

            for row in data:
                if isinstance(row, str):
                    hasil_preprocess.append(row.split(','))
                else:
                    hasil_preprocess.append(row)

            document_word_vectors = []  # list untuk menyimpan pasangan kata dan vector untuk setiap dokumen
            document_vectors = []  # list untuk menyimpan vector rata-rata dari setiap dokumen

            for doc in hasil_preprocess:
                if isinstance(doc, list):
                    word_vectors = []
                    for word in doc:
                        if word in model_w2v.wv:
                            word_vector = model_w2v.wv[word]  # mendapatkan vector kata dari model
                            word_vectors.append((word, word_vector))  # menambahkan pasangan kata dan vector ke dalam list
                        else:
                            # Jika kata tidak ada dalam model, tambahkan vektor nol
                            word_vector = np.zeros(model_w2v.vector_size)
                            word_vectors.append((word, word_vector))

                    if word_vectors:
                        total_vector = 0
                        for pasangan_kata_vector in word_vectors:
                            vector = pasangan_kata_vector[1]
                            total_vector += vector

                        doc_vector = total_vector / len(word_vectors)
                        document_word_vectors.append(word_vectors)  # menambahkan pasangan kata dan vector untuk dokumen saat ini
                        document_vectors.append(doc_vector)  # menambahkan vector rata-rata ke dalam list
                    else:
                        document_vectors.append(None)

            # st.write('Vector Kata')
            # for i in document_word_vectors:
            #     st.write(i)

            # st.write('Vector Dokumen')
            if document_vectors is not None:
                document_vectors_df = pd.DataFrame(document_vectors)
                # st.write(document_vectors_df)
            # else:
            #     data = {i+1: [0] for i in range(100)}
            #     document_vectors_df = pd.DataFrame(data)
            #     # document_vectors_df
            #     st.write(document_vectors_df)

            #klasifikasi menggunakan SVM
            with open('Model/Skenario 2 _ w7 _ c10g1 rbf.pkl','rb') as r:
                model_svm = pickle.load(r)

            pred = model_svm.predict(document_vectors_df)
            if pred == 0:
                    st.write(f'Ulasan "{teks_input}" memiliki <span style="background-color:#fb4c4c; padding: 5px; border-radius: 5px; color:white;">**Sentimen Negatif**</span>', unsafe_allow_html=True)
            if pred == 1:
                st.write(f'Ulasan "{teks_input}" memiliki <span style="background-color:#1e81b0; padding: 5px; border-radius: 5px; color:white;">**Sentimen Positif**</span>', unsafe_allow_html=True)

        else :
            st.warning('Anda Belum Masukkan Teks', icon="⚠️")
