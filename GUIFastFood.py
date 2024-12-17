import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Muat model dan PCA
kmeans = joblib.load('kmeans_model.pkl')
pca = joblib.load('pca_model.pkl')

st.title("Prediksi Cluster Berdasarkan Nutrisi Fast Food")

# This is The Header
st.header("Kelompok 6 Kelas D:", divider="green")

# This is The Sub-header
st.subheader("1. Muhammad Naufal Izzudin (24060122120018)")
st.subheader("2. Tiara Putri Wibowo (24060122120026)")
st.subheader("3. Fendi Ardianto (24060122130077)")
st.subheader("4. Farrel Ardana Jati (24060122140165)")

st.header("Impor Dataframe", divider="gray")
df = pd.read_csv("fastfood.csv")
st.dataframe(df)

st.header("Pembagian Cluster ")
st.text("(Berdasarkan data yang sudah dinormalisasi MinMax)")
st.subheader("Cluster 0: Everyday Meals")
st.markdown("""
Karakteristik:

Calories (0.285), Sodium (0.330), Total Fat (0.254): Relatif rendah, menunjukkan makanan ini cocok untuk mereka yang ingin menjaga asupan kalori dan lemak tanpa terlalu membatasi.

Fiber (0.332): Moderat, mendukung kesehatan pencernaan.

Protein (0.276): Cukup baik untuk memenuhi kebutuhan protein dasar.

Vitamin A dan C (0.292, 0.320): Sedang, cocok untuk kebutuhan nutrisi standar.

Calcium (0.414): Moderat, mendukung kesehatan tulang.


Makanan di cluster ini cocok untuk konsumen yang menginginkan makanan dengan komposisi nutrisi yang seimbang tetapi tidak terlalu rendah atau tinggi dalam satu aspek.
""")

st.subheader("Cluster 1: Energy Booster")
st.markdown("""
Karakteristik:

Calories (0.567), Sodium (0.628), Total Carb (0.738): Tinggi, menunjukkan makanan ini kaya kalori dan karbohidrat, cocok untuk mereka yang membutuhkan energi lebih banyak.

Fiber (0.774): Sangat tinggi, mendukung kesehatan pencernaan dengan sangat baik.

Sugar (0.613): Tinggi, cocok untuk konsumen yang menyukai makanan manis.

Vitamin C dan Calcium (0.726, 0.836): Tinggi, mendukung kesehatan secara keseluruhan.

Protein (0.484): Tinggi, cocok untuk memenuhi kebutuhan protein lebih tinggi.


Makanan dalam cluster ini cocok untuk mereka yang membutuhkan asupan energi dan nutrisi lebih tinggi, seperti atlet atau individu dengan kebutuhan kalori tinggi.
""")

st.subheader("Cluster 2: Protein-Rich Meals")
st.markdown("""
Karakteristik:

Total Fat (0.652), Cholesterol (0.669), Protein (0.643): Tinggi, menunjukkan makanan ini cocok untuk diet yang fokus pada lemak dan protein, seperti diet keto.

Calories (0.640): Tinggi, mendukung kebutuhan energi.

Fiber dan Sugar (0.324, 0.519): Moderat.

Calcium (0.526): Sedang, cukup mendukung kebutuhan tulang.


Makanan ini cocok untuk konsumen yang fokus pada asupan protein dan lemak tinggi, seperti mereka yang menjalani pola makan tinggi lemak untuk kebutuhan tertentu.
""")

st.subheader("Cluster 3: Low-Calorie Meals")
st.markdown("""
Karakteristik:

Calories (0.230), Total Fat (0.250), Sodium (0.253): Rendah, menunjukkan makanan ini cocok untuk mereka yang ingin menjaga berat badan atau memilih makanan rendah kalori.

Vitamin A dan C (1.000, 0.962): Sangat tinggi, mendukung kebutuhan mikronutrien.

Fiber (0.427): Moderat.

Protein (0.265): Relatif rendah.

Calcium (0.329): Rendah.


Cluster ini cocok untuk makanan rendah kalori tetapi tinggi mikronutrien, cocok untuk konsumen yang memprioritaskan vitamin dalam pola makan mereka, seperti mereka yang ingin meningkatkan asupan vitamin tanpa meningkatkan kalori.
""")

st.header("Prediksi Makanan Anda!", divider="gray")
# Input data nutrisi untuk setiap fitur
name = st.text_input('Nama Makanan')
# Mengatur layout menjadi dua kolom
col1, col2 = st.columns(2)

# Input fields di kolom pertama
with col1:
    calories = st.number_input('Calories (kkal)', value=0.5, min_value=0.0, max_value=2430.0)
    total_fat = st.number_input('Total Fat (g)', value=0.3, min_value=0.0, max_value=141.0)
    cholesterol = st.number_input('Cholesterol (mg)', value=0.2, min_value=0.0, max_value=805.0)
    sodium = st.number_input('Sodium (mg)', value=0.7, min_value=0.0, max_value=6080.0)
    total_carb = st.number_input('Total Carbohidrat (g)', value=0.4, min_value=0.0, max_value=156.0)

# Input fields di kolom kedua
with col2:
    fiber = st.number_input('Fiber (g)', value=0.1, min_value=0.0, max_value=17.0)
    sugar = st.number_input('Sugar (g)', value=0.2, min_value=0.0, max_value=87.0)
    protein = st.number_input('Protein (g)', value=0.5, min_value=0.0, max_value=186.0)
    vit_a = st.number_input('Vitamin A (µg)', value=0.8, min_value=0.0, max_value=180.0)
    vit_c = st.number_input('Vitamin C (mg)', value=0.6, min_value=0.0, max_value=400.0)
    calcium = st.number_input('Calcium (%DV)', value=0.4, min_value=0.0, max_value=290.0)


# Input data sebagai array
input_data = np.array([calories, total_fat, cholesterol, sodium, total_carb, fiber, sugar, protein, vit_a, vit_c, calcium]).reshape(1, -1)

if st.button("Prediksi"):
    try:
        # Transformasi PCA (hanya menggunakan fitur yang relevan)
        pca_features = input_data[:, [1, 2, 3, 4, 7, 0]]  # Kolom yang dipilih untuk PCA
        pca_transformed = pca.transform(pca_features)

        # Gabungkan data PCA dengan fitur tambahan
        additional_features = input_data[:, [5, 6, 8, 9, 10]]  # Fitur tambahan
        data_combined = np.hstack((pca_transformed, additional_features))  # Gabungkan sebagai array 2D

        # Prediksi cluster dengan KMeans
        prediction = kmeans.predict(data_combined)

        # Interpretasi cluster
        cluster_labels = {
            0: "Everyday Meals",
            1: "Energy Booster",
            2: "Protein-Rich Meals",
            3: "Low-Calorie Meals"
        }
        st.success(f"{name} tergolong makanan dengan kategori {prediction[0]} - {cluster_labels.get(prediction[0], 'Unknown')}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")





# # This is the Title
# st.title("Tugas Besar Kelompok X")

# # This is The Header
# st.header("Dikerjakan oleh", divider="green")

# # This is The Sub-header
# st.subheader("1. Axelliano")
# st.subheader("2. Kevin ")

# # Impor Dataframe
# st.header("Impor Dataframe", divider="gray")
# df = pd.read_csv("iris.csv")
# st.dataframe(df)

# # Create sub-tabs for each feature using st.radio
# feature = st.radio("Select Feature to Display", ('Sepal Width',
#                    'Sepal Length', 'Petal Width', 'Petal Length'))

# # Based on the feature selected, display the corresponding bar chart
# if feature == 'Sepal Width':
#     st.bar_chart(df['sepal_width'])
# elif feature == 'Sepal Length':
#     st.bar_chart(df['sepal_length'])
# elif feature == 'Petal Width':
#     st.bar_chart(df['petal_width'])
# elif feature == 'Petal Length':
#     st.bar_chart(df['petal_length'])

# st.header("Outlook")
# # Tab Version
# tab1, tab2, tab3, tab4 = st.tabs(
#     ["Sepal Length", "Sepal Width", "Petal Width", "Petal Length"])
# with tab1:
#     st.subheader("Sepal Length")
#     st.bar_chart(df['sepal_width'])
# with tab2:
#     st.bar_chart(df['sepal_length'])
# with tab3:
#     st.bar_chart(df['petal_width'])
# with tab4:
#     st.bar_chart(df["petal_length"])

# st.title("Iris Dataset Correlation Heatmap")

# # Create a heatmap of the correlation matrix using Seaborn
# corr_matrix = df.iloc[:, :4].corr()

# # Set up the matplotlib figure
# plt.figure(figsize=(10, 8))

# # Create a heatmap
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
#             fmt='.2f', linewidths=0.5)

# # Display the heatmap in Streamlit
# st.pyplot(plt)

# # train_model

# label_encoder = LabelEncoder()
# df['species'] = label_encoder.fit_transform(df['species'])

# # Step 2: Normalization using StandardScaler
# scaler = StandardScaler()

# # Apply scaler to numeric columns only (price and rating)
# df.iloc[:, :4] = scaler.fit_transform(df.iloc[:, :4])


# X = df.drop(columns=['species'])
# y = df['species']

# # Split the dataset into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# # Step 4: Train a Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# # Step 5: Make Predictions and Evaluate
# y_pred = rf_classifier.predict(X_test)

# # Evaluate the model (Accuracy Score)
# accuracy = accuracy_score(y_test, y_pred)
# st.write("Accuracy of Random Forest model:", accuracy)

# # Optionally, view the predicted vs actual values
# st.write("Predicted values:", y_pred)
# st.write("Actual values:", y_test.values)

# # Predict The new
# st.header("PREDICT NEW VALUES")
# input_sepal_length = st.number_input(
#     "sepal_length", min_value=0.1, max_value=7.0, step=0.01)
# input_sepal_width = st.number_input(
#     "sepal_width", min_value=0.1, max_value=7.0, step=0.01
# )
# input_petal_length = st.number_input(
#     "petal_length", min_value=0.1, max_value=7.0, step=0.01)
# input_petal_width = st.number_input(
#     "petal_width", min_value=0.1, max_value=7.0, step=0.01)
# predict_button = st.button("Predict")

# if predict_button:
#     input_data = pd.DataFrame({
#         'sepal_length': [input_sepal_length],
#         'sepal_width': [input_sepal_width],
#         'petal_length': [input_petal_length],
#         'petal_width': [input_petal_width],
#     })
#     st.dataframe(input_data)
#     input_data_scaled = scaler.transform(input_data)
#     prediction = rf_classifier.predict(input_data)
#     predicted_species = label_encoder.inverse_transform(prediction)
#     st.subheader(f"Predicted Species: {predicted_species[0]}")
# # st.markdown("*Streamlit* is **really** ***cool***.")
# # st.markdown('''
# #     :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
# #     :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')
# # st.markdown("Here's a bouquet &mdash;\
# #             :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

# # multi = '''If you end a line with two spaces,
# # a soft return is used for the next line.

# # Two (or more) newline characters in a row will result in a hard return.
# # '''
# # st.markdown(multi)

# # md = st.text_area('Type in your markdown string (without outer quotes)',
# #                   "Happy Streamlit-ing! :balloon:")

# # st.code(f"""
# # import streamlit as st

# # st.markdown('''{md}''')
# # """)

# # st.markdown(md)

# option = st.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone"),
# )

# st.write("You selected:", option)
