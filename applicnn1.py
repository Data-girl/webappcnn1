# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:03:53 2020

@author: Bellemiss972
"""


##CHARGEMENT DES LIBRAIRIES
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

##CHARGEMENT DU MODELE
model = tf.keras.models.load_model('cancercnn.h5')

##FONCTION DE PRÉDICTION

def import_et_predict(image_data, model):
    
    size=(201, 159)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image=np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = np.expand_dims(image, axis = 0)
    img_reshape = image[np.newaxis,...,np.newaxis]

    prediction = model.predict(img_reshape)
        
    return prediction

model = tf.keras.models.load_model('cancercnn.h5')

## FONCTION PRINCIPALE
def run():
    
    html_template = """
    <div style = "background-color : #85c99b; padding:15px">
    <h2 style="color : white; text-align :center ; ">Interface de détection du cancer des poumons</h2>"""
    st.markdown(html_template, unsafe_allow_html=True)
    st.write("")  
    
    #Création et personnalisation de la sidebar
    
    #Logo de l'entreprise
    logo = Image.open('logo.png')
    
    # st.sidebar.image(logo,use_column_width=True)
    
    st.sidebar.info("Cette application web a été réalisée à des fins de démonstration")
    
    #Illustration médicale
    image2=Image.open('scanner2.jpg')
    st.sidebar.image(image2, use_column_width=True)
    
    #Personnalisation page centrale
    st.sidebar.success("Vous souhaitez concevoir votre propre interface de prédiction afin de détecter les cas de cancer des poumons ?)
    
    st.write("Cette interface a pour objectif de faciliter la détection du cancer des poumons. Elle a été conçue à partir d'une base de données de 6000 références de scanner des poumons (sains et cancéreux), et offre une **précision de 99%** dans l'identification des patients atteints par la maladie. Le modèle de prédiction repose sur un réseau de neurones à reconnaissance d'image, aussi connu sous l'appellation de réseaux de neurones convolutifs (CNN).")
    
    st.write(""" [🟢 - Cliquez ici pour réaliser un test en téléchargeant une image de poumons présentant un état sain](https://drive.google.com/uc?export=download&id=1Muzi-Fzf0z4B81Tcpd_5gvDwkQkl40GM)""")
    st.write(""" [🔴 - Cliquez ici pour réaliser un test en téléchargeant une image de poumons présentant un état cancéreux](https://drive.google.com/uc?export=download&id=1Df7eDWDR1hxsTZLGpLYoZbV6BC0uIV8w)""")
         
    file=st.file_uploader(" Veuillez charger une des deux images téléchargées précédemment en cliquant sur ''browse files'' ⬇️ ",type=['jpeg','png','jpg'])
    
    #Traitement de l'image chargée
    if file is not None :
        
        image=Image.open(file)
        st.image (image, use_column_width=True)
        prediction= import_et_predict(image,model)
        
        #Prédiction et légence explicative
        if prediction > 0.5:
            
            st.success("""Les résultats indiquent un **état sain des poumons** chez le patient""")
        else: 
            st.error("""Les résultats indiquent un **état cancéreux des poumons** chez le patient""")
        
        np.set_printoptions(suppress=True)
        st.write("""Le **seuil de probabilité** est estimé à {}""".format(prediction))
        st.warning("""Plus le seuil de probabilité **se rapproche de 0**, plus le risque de cancer est élevé. À l'inverse, plus il **s'approche de 1**, plus le risque de cancer du poumon est faible.""")

    
if __name__ =='__main__' : run()
