import streamlit as st
import pandas as pd

basari = pd.read_csv("stres.csv")
basari["student_id"] = basari["student_id"].astype(int)
basari["true_ratio_cat"] = basari["true_ratio_cat"].astype(int)
nos = basari["student_id"]
# stres = stres.apply(lambda x: "öğrenci no: " + str(int(x["student_id"])) + " - başarı skoru: " + str(int(x["true_ratio_cat"])), axis=1)

st.title("Student form")


with st.form("form"):
    disturbed_level = st.slider(
        "Son bir ay içinde, beklenmedik bir şeylerin olması nedeniyle ne sıklıkta rahatsızlık duydunuz?",
        0,
        4,
        0,
    )

    control_loss = st.slider(
        "Son bir ay içinde ne sıklıkta, yaşamınızdaki önemli şeyleri kontrol edemediğinizi hissettiniz?",
        0,
        4,
        0,
    )

    stress_level = st.slider(
        "Son bir ay içinde kendinizi ne sıklıkta, gergin ve stresli hissettiniz?",
        0,
        4,
        0,
    )

    coping_success = st.slider(
        "Son bir ay içinde, yaşamınızdaki can sıkıcı durumlarla ne sıklıkta başarılı bir biçimde baş ettiniz?",
        0,
        4,
        0,
    )

    change_management = st.slider(
        "Son bir ay içinde ne sıklıkta, yaşamınızda meydana gelen önemli değişikliklerle etkili bir biçimde başa çıktığınızı hissettiniz?",
        0,
        4,
        0,
    )

    problem_solving = st.slider(
        "Son bir ay içinde ne sıklıkta, kişisel sorunlarınızla baş etme yeteneğinizden emin oldunuz?",
        0,
        4,
        0,
    )

    life_satisfaction = st.slider(
        "Son bir ay içinde ne sıklıkta, işlerin istediğiniz gibi gittiğini hissettiniz?",
        0,
        4,
        0,
    )

    overwhelmed_feeling = st.slider(
        "Son bir ay içinde ne sıklıkta, yapmak zorunda olduğunuz her şeyin üstesinden gelemeyeceğinizi düşündünüz?",
        0,
        4,
        0,
    )

    event_control = st.slider(
        "Son bir ay içinde yaşamınızdaki rahatsız edici olayları ne sıklıkta kontrol edebildiniz?",
        0,
        4,
        0,
    )

    life_dominance = st.slider(
        "Son bir ay içinde ne sıklıkta, yaşamınızdaki olaylara hakim olduğunuzu hissettiniz?",
        0,
        4,
        0,
    )

    anger_level = st.slider(
        "Son bir ay içinde, kontrolünüz dışında gerçekleşen şeylerden dolayı ne sıklıkta öfkelendiniz?",
        0,
        4,
        0,
    )

    overthinking = st.slider(
        "Son bir ay içinde ne sıklıkta, üstesinden gelmek zorunda olduğunuz şeyler üzerinde düşündünüz?",
        0,
        4,
        0,
    )

    time_management = st.slider(
        "Zamanınızı nasıl geçirdiğinizi son bir ay içinde ne sıklıkta kontrol edebildiniz?",
        0,
        4,
        0,
    )

    difficulty_overload = st.slider(
        "Son bir ay içinde ne sıklıkta, güçlüklerin, üstesinden gelemeyeceğiniz kadar çoğaldığını hissettiniz?",
        0,
        4,
        0,
    )

    student_selection = st.selectbox("Öğrenci/Başarı skoru seçimi", options=nos, format_func=lambda x: "Öğrenci no: " + str(x) + " - Başarı skoru: " + str(basari[basari["student_id"] == x]["true_ratio_cat"].iloc[0]))

    reason = st.text_area("Anlatmak istediğin bir şey var mı?")
    submitted = st.form_submit_button("Submit")

    if submitted:
        import openai
        import joblib

        with open("random_forest_model.pkl", "rb") as f:
            random_forest = joblib.load(f)

        data = {
            "disturbed_level": disturbed_level,
            "control_loss": control_loss,
            "stress_level": stress_level,
            "coping_success": coping_success,
            "change_management": change_management,
            "problem_solving": problem_solving,
            "life_satisfaction": life_satisfaction,
            "overwhelmed_feeling": overwhelmed_feeling,
            "event_control": event_control,
            "life_dominance": life_dominance,
            "anger_level": anger_level,
            "overthinking": overthinking,
            "time_management": time_management,
            "difficulty_overload": difficulty_overload,
        }

        stress_scores = random_forest.predict(
            [
                [
                    data["disturbed_level"],
                    data["control_loss"],
                    data["stress_level"],
                    data["coping_success"],
                    data["change_management"],
                    data["problem_solving"],
                    data["life_satisfaction"],
                    data["overwhelmed_feeling"],
                    data["event_control"],
                    data["life_dominance"],
                    data["anger_level"],
                    data["overthinking"],
                    data["time_management"],
                    data["difficulty_overload"],
                ]
            ]
        )

        basari_score = basari[basari["student_id"] == student_selection]["true_ratio_cat"].iloc[0]

        system_prompt = """\
Sen bir danışman sekreterisin. Aşağıdaki öğrenci metni ve ek meta veriler üzerinden öğrencinin gününü, yaptığı aktiviteleri ve duygusal durumunu 1-2 kısa cümle ile özetle.
Stres skoru: 0-56 arası değişebilen bir değer. Eğer 28'den büyük ise öğrencinin stresli olduğunu söyleyebiliriz.
Başarı skoru: 0-5 arası değişebilen bir değer. 5 en yüksek başarıyı, 0 en düşük başarıyı temsil ediyor. Bu skor sınavlardan aldığı puana göre hesaplanıyor.

Örnek 1:
Öğrenci metni: Bugün sabah erken kalkıp ders çalıştım, öğleden sonra arkadaşlarımla buluştum fakat akşam sınavım kötü geçti.
Meta veriler: {ruh hali: "üzgün"}
Özet: Öğrenci, gününü genel olarak verimli geçirirken sınav sonucu nedeniyle moral kaybı yaşamış

Örnek 2:
Öğrenci metni: Bugün spor yaptım, kütüphanede yoğun çalıştım ve akşam yeni bir hobi denedim.
Meta veriler: {ruh hali: "mutlu"}
Özet: Öğrenci, enerjik ve olumlu bir gün geçirmiş.

Örnek 3:
Öğrenci metni: Sınavlar yaklaşıyor ve çalışmam gereken konular birikmiş durumda. Ne kadar uğraşırsam uğraşayım, sanki yeterince ilerleyemiyorum. Sürekli bir yetişememe hissi var içimde. Okulda herkes başarılı görünürken ben neden bu kadar zorlanıyorum bilmiyorum. Günler geçtikçe kaygım artıyor, ama elimden geleni yapmaya çalışıyorum.
Meta veriler: {ruh hali: "yorgun", "mutsuz"}
Özet: Öğrenci, çıkmazda hissediyor. Kendini diğer insanlarla karşılaştırınca kaygılanıyor.

Şimdi, verilen öğrenci metni ve meta veriler doğrultusunda benzer şekilde özet çıkar."""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"Öğrenci metni: {reason}\n"
                        f"Meta veriler: {{stres skoru: {stress_scores[0]}, başarı skoru: {str(basari_score)}}}\n"
                        "Özet:"
                    ),
                },
            ],
            temperature=0,
        )

        st.divider()

        st.success(
            f"""\
Rapor danışmana başarıyla gönderildi.\n
Hesaplanan stres seviyesi: {'Stresli' if int(stress_scores[0]) > 28 else 'Stresli değil'} (skor: {stress_scores[0]})\n
Danışmana gönderilen bildirim: {response.choices[0].message.content}"""
        )
