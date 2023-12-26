import streamlit as st
from PIL import Image
import pickle


model = pickle.load(open('./Model/ML_Model.pkl', 'rb'))

def run():
    img1 = Image.open('bank.png')
    img1 = img1.resize((156,145))
    st.image(img1,use_column_width=False)
    st.title("MiuulBank Kredi Sistemi")

    ## Account No
    account_no = st.text_input(' HESAP NO')

    ## Full Name
    fn = st.text_input('İSİM SOYİSİM')

    ## For gender
    gen_display = ('KADIN','ERKEK')
    gen_options = list(range(len(gen_display)))
    gen = st.selectbox("CİNSİYET",gen_options, format_func=lambda x: gen_display[x])

    ## For Marital Status
    mar_display = ('HAYIR','EVET')
    mar_options = list(range(len(mar_display)))
    mar = st.selectbox("MEDENİ HAL EVLİ", mar_options, format_func=lambda x: mar_display[x])

    ## No of dependets
    dep_display = ('0','1','2','2 ÜSTÜ')
    dep_options = list(range(len(dep_display)))
    dep = st.selectbox("ÇOCUK SAYISI",  dep_options, format_func=lambda x: dep_display[x])

    ## For edu
    edu_display = ('OKUL MEZUN','OKUL MEZUN DEĞİL')
    edu_options = list(range(len(edu_display)))
    edu = st.selectbox("EĞİTİM DURUMU",edu_options, format_func=lambda x: edu_display[x])

    ## For emp status
    emp_display = ('MESLEK','KENDİ İŞİ VAR')
    emp_options = list(range(len(emp_display)))
    emp = st.selectbox("İŞ DURUMU",emp_options, format_func=lambda x: emp_display[x])

    ## For Property status
    prop_display = ('KöY','YARIKENTSEL','BÜYÜKŞEHİR')
    prop_options = list(range(len(prop_display)))
    prop = st.selectbox("",prop_options, format_func=lambda x: prop_display[x])

    ## For Credit Score
    cred_display = (' 300 - 500',' 500 ÜSTÜ')
    cred_options = list(range(len(cred_display)))
    cred = st.selectbox("KREDİ SKORU",cred_options, format_func=lambda x: cred_display[x])
    
    ## Applicant Monthly Income
    mon_income = st.number_input("BAŞVURU SAHİBİNİN AYLIK GELİRİ ($)",value=0)
    
    ## Co-Applicant Monthly Income
    co_mon_income = st.number_input("KEFİLİN AYLIK GELİRİ ($)",value=0)

    ## Loan AMount
    loan_amt = st.number_input("KREDİ MİKTARI",value=0)

    ## loan duration
    dur_display = ['2','6','8','12','16 ']
    dur_options = range(len(dur_display))
    dur = st.selectbox("KREDİ SÜRESİ(AY)",dur_options, format_func=lambda x: dur_display[x])

    if st.button("GİRİŞ"):
        duration = 0
        if dur == 0:
            duration = 60
        if dur == 1:
            duration = 180
        if dur == 2:
            duration = 240
        if dur == 3:
            duration = 360
        if dur == 4:
            duration = 480
        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
        print(features)
        prediction = model.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        if ans == 0:
            st.error(
                "MERHABALAR: " + fn +" || "
                "HESAPNO: "+account_no +' || '
                'HESAPLARA GÖRE BANKAMIZDAN KREDİ ALAMIYORSUNUZ'
            )
        else:
            st.success(
                "MERHABALAR: " + fn +" || "
                "HESAPNO: "+account_no +' || '
                ' HESAPLARA GÖRE BANKAMIZDAN KREDİ ALABİLİRSİNİZ'
            )

run()


