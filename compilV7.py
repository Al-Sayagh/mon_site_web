# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import sys
import streamlit as st
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Installer fuzzywuzzy et python-Levenshtein si non installé
try:
    from fuzzywuzzy import process
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy"])
    from fuzzywuzzy import process

try:
    import Levenshtein
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-Levenshtein"])
    import Levenshtein

# Titre et présentation de l'application
st.title("Exploration Avancée des Données et Système de Recommandation de Films")
st.write("Bienvenue dans cette application interactive Streamlit qui permet de charger et d'explorer vos fichiers CSV, ainsi que de générer des recommandations personnalisées de films pour vos clients.")

# Charger les données
file_path = r'C:\Users\724me\OneDrive\Desktop\dossier ANA\df_demonstration.csv'

@st.cache_data(show_spinner=False)
def load_data(file_path):
    return pd.read_csv(file_path)

df_short = load_data(file_path)

# Menu latéral pour choisir la fonctionnalité
option = st.sidebar.radio("Choisissez une fonctionnalité :", [
    "Analyse des Données",
    "Recommandations Personnalisées de Films",
    "Évaluation des Modèles de Recommandation",
    "Évaluation des Modèles de Recommandation Rapide"
])

if option == "Analyse des Données":
    # Affichage des informations générales
    st.subheader("Résumé des Données Chargées")
    st.write(f"Le fichier contient **{df_short.shape[0]} lignes** et **{df_short.shape[1]} colonnes**.")
    st.dataframe(df_short, height=300)

    # Afficher des métriques intéressantes sur les données
    st.subheader("Métriques Clés des Données")
    df_short['nombre_votes_imdb'] = pd.to_numeric(df_short['nombre_votes_imdb'], errors='coerce').fillna(0).astype(int)
    total_votes = df_short['nombre_votes_imdb'].sum()
    avg_votes_per_user = df_short[df_short['score_pertinence'].notna()].groupby('nom_utilisateur').size().mean()
    num_unique_users = df_short['nom_utilisateur'].nunique()
    num_movies = df_short['titre_film'].nunique()
    top_voted_movies = df_short[['titre_film', 'nombre_votes_imdb']].drop_duplicates().sort_values(by='nombre_votes_imdb', ascending=False).head(10)

    st.write(f"- **Nombre total de votes IMDB**: {total_votes:,}")
    st.write(f"- **Nombre moyen de votes par utilisateur**: {avg_votes_per_user:.2f}")
    st.write(f"- **Nombre total d'utilisateurs ayant participé**: {num_unique_users}")
    st.write(f"- **Nombre total de films dans la base**: {num_movies}")
    st.write("- **Top 10 des films les plus votés**:")
    st.dataframe(top_voted_movies, use_container_width=True)

    # Options interactives pour explorer les données
    st.subheader("Exploration Interactive des Données")
    columns_to_display = st.multiselect("Sélectionnez les colonnes à afficher :", df_short.columns.tolist(), default=df_short.columns.tolist())
    if columns_to_display:
        st.write("Aperçu des colonnes sélectionnées :")
        st.dataframe(df_short[columns_to_display], height=300)
    else:
        st.warning("Veuillez sélectionner au moins une colonne pour afficher les données.")

    # Afficher les statistiques descriptives
    if st.checkbox("Afficher les statistiques descriptives"):
        st.subheader("Statistiques Descriptives des Données")
        st.write(df_short.describe())

    # Recherche dynamique dans une colonne
    st.subheader("Recherche Dynamique dans les Données")
    search_column = st.selectbox("Choisissez une colonne pour effectuer une recherche :", df_short.columns)
    search_value = st.text_input(f"Entrez une valeur à rechercher dans la colonne '{search_column}' :")
    if search_value:
        filtered_data = df_short[df_short[search_column].astype(str).str.contains(search_value, case=False, na=False)]
        st.write(f"Résultats de la recherche pour **'{search_value}'** dans la colonne **'{search_column}'** :")
        st.dataframe(filtered_data, height=300)
        st.write(f"Nombre de résultats trouvés : {filtered_data.shape[0]}")
    else:
        st.info("Entrez une valeur pour lancer une recherche.")

elif option == "Recommandations Personnalisées de Films":
    # Choix des algorithmes de recommandation
    st.subheader("Choix de l'Algorithme de Recommandation")
    choix_principal = st.selectbox("Choisissez une méthode de filtrage :", [
        "Filtrage Collaboratif : approche mémoire",
        "Filtrage Collaboratif : approche modèle",
        "Filtrage basé sur le contenu"
    ])

    if choix_principal == "Filtrage Collaboratif : approche mémoire":
        sous_choix = st.radio("Choisissez une approche :", ["User-based", "Item-based"])

        # Charger les données
        df = load_data(file_path)

        # Création de la Matrice de Notations
        mat_ratings = df.pivot_table(index='id_utilisateur', columns='titre_film', values='score_pertinence').fillna(0)

        # Conversion au Format Sparse
        sparse_ratings = csr_matrix(mat_ratings.values)

        # Calcul des similarités
        user_similarity_matrix = cosine_similarity(sparse_ratings)
        user_similarity_df = pd.DataFrame(user_similarity_matrix, index=mat_ratings.index, columns=mat_ratings.index)

        item_similarity_matrix = cosine_similarity(sparse_ratings.T)
        item_similarity_df = pd.DataFrame(item_similarity_matrix, index=mat_ratings.columns, columns=mat_ratings.columns)

        # Fonctions de prédiction
        def pred_user(mat_ratings, user_similarity, k, user_id):
            user_ratings = mat_ratings.loc[user_id]
            to_predict = user_ratings[user_ratings == 0]
            similar_users = user_similarity.loc[user_id].sort_values(ascending=False).iloc[1:k + 1]
            norm = np.sum(np.abs(similar_users))

            for i in to_predict.index:
                ratings = mat_ratings[i].loc[similar_users.index]
                scalar_prod = np.dot(similar_users, ratings)
                to_predict[i] = scalar_prod / (norm + 1e-8)

            return to_predict

        def pred_item(mat_ratings, item_similarity, k, user_id):
            user_ratings = mat_ratings.loc[user_id]
            to_predict = user_ratings[user_ratings == 0]

            for i in to_predict.index:
                similar_items = item_similarity[i].sort_values(ascending=False)[1:k + 1]
                norm = np.sum(np.abs(similar_items))
                ratings = mat_ratings.loc[user_id, similar_items.index]
                scalar_prod = np.dot(ratings, similar_items)
                to_predict[i] = scalar_prod / (norm + 1e-8)

            return to_predict

        # Sélection de l'utilisateur via un menu déroulant
        user_id = st.selectbox("Choisissez un utilisateur :", df['nom_utilisateur'].unique())

        if user_id:
            user_id_value = df[df['nom_utilisateur'] == user_id]['id_utilisateur'].iloc[0]
            if sous_choix == "User-based":
                top_10_user_based = pred_user(mat_ratings, user_similarity_df, k=5, user_id=user_id_value).sort_values(ascending=False).head(10)
                st.write(f"💖 Top 10 des films recommandés pour {user_id} (User-based) :")
                for i, (film, score) in enumerate(top_10_user_based.items(), 1):
                    etoiles = "⭐" * (int(score) // 20)
                    st.markdown(f"{i}. {film}  \nScore de pertinence prédit : {score:.2f} {etoiles}")
            elif sous_choix == "Item-based":
                top_10_item_based = pred_item(mat_ratings, item_similarity_df, k=5, user_id=user_id_value).sort_values(ascending=False).head(10)
                st.write(f"🎬 Top 10 des films recommandés pour {user_id} (Item-based) :")
                for i, (film, score) in enumerate(top_10_item_based.items(), 1):
                    etoiles = "⭐" * (int(score) // 20)
                    st.markdown(f"{i}. {film}  \nScore de pertinence prédit : {score:.2f} {etoiles}")  

        # Affichage des calculs intermédiaires pour le débogage
        if st.checkbox("Afficher les matrices de similarité (pour le débogage) :"):
            st.write("Matrice de similarité entre utilisateurs :")
            st.dataframe(user_similarity_df)
            st.write("Matrice de similarité entre items :")
            st.dataframe(item_similarity_df)




# Filtrage basé sur le contenu avec TF-IDF
    @st.cache_resource(show_spinner=False)
    def compute_tfidf_matrix(df):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=10, max_features=1000)
        df['description'] = df['description'].fillna('')
        return tfidf_vectorizer, tfidf_vectorizer.fit_transform(df['description'])

    tfidf_vectorizer, tfidf_matrix = compute_tfidf_matrix(df_short)

    def get_best_match(input_text, column_data):
        matches = process.extract(input_text, column_data, limit=3)
        return matches

    def recommend_by_content(best_match, search_by='titre_film', n_recommendations=5, min_score=0.1, metric='cosine'):
        if search_by == 'noms_acteurs':
            filtered_df = df_short[df_short['noms_acteurs'].str.contains(best_match, case=False, na=False)]
        else:
            filtered_df = df_short[df_short[search_by] == best_match]
        if filtered_df.empty:
            return pd.DataFrame(), None
        idx = filtered_df.index[0]
        if metric == 'cosine':
            similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            similar_indices = similarities.argsort()[::-1]
        elif metric == 'euclidean':
            distances = euclidean_distances(tfidf_matrix[idx], tfidf_matrix).flatten()
            similar_indices = distances.argsort()
            similarities = 1 / (1 + distances)
        similar_scores = similarities[similar_indices]
        filtered_indices = []
        seen_titles = set()
        for i, score in zip(similar_indices, similar_scores):
            if score >= min_score and df_short.iloc[i]['titre_film'] != best_match:
                titre = df_short.iloc[i]['titre_film']
                if titre not in seen_titles:
                    filtered_indices.append(i)
                    seen_titles.add(titre)
                    if len(filtered_indices) >= n_recommendations:
                        break
        filtered_scores = [similarities[i] for i in filtered_indices]
        recommendations = df_short.iloc[filtered_indices].copy()
        recommendations['similarité (%)'] = [round(score * 100, 2) for score in filtered_scores]
        selected_film_details = df_short.iloc[idx]
        return recommendations[['titre_film', 'noms_acteurs', 'realisateurs_principaux', 'score_pertinence', 'similarité (%)']], selected_film_details

    if choix_principal == "Filtrage basé sur le contenu":
        search_option = st.selectbox("🔍 Rechercher par :", ["titre_film", "noms_acteurs", "realisateurs_principaux"], index=0)
        user_input = st.text_input(f"**Entrez un(e) {search_option.replace('_', ' ')} :**", key="user_input", autocomplete='off')
        n_recommendations = st.slider("🌜 Nombre de recommandations :", min_value=1, max_value=20, value=5)
        min_score = st.slider("🌋 Seuil de similarité minimale :", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        min_user_rating = st.slider("📈 Score de pertinence minimale :", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        metric_option = st.radio("🔢 Choisissez la méthode de similarité :", ["cosine", "euclidean"], index=0)
        if user_input:
            matches = get_best_match(user_input, df_short[search_option].dropna().unique() if search_option != 'noms_acteurs' else df_short['noms_acteurs'].str.split(',').explode().str.strip().unique())
            if matches:
                match_options = [match[0] for match in matches]
                best_match = st.selectbox("⚙️ Est-ce que l'un de ces titres correspond à votre recherche ?", match_options, key="best_match", index=0)
                if best_match:
                    recommendations, selected_film_details = recommend_by_content(best_match, search_by=search_option, n_recommendations=n_recommendations, min_score=min_score, metric=metric_option)
                    if selected_film_details is not None:
                        st.write(f"### 📜 Détails du film sélectionné : **{best_match}**")
                        st.write(selected_film_details[['titre_film', 'description', 'noms_acteurs', 'realisateurs_principaux', 'score_pertinence']])
                    if not recommendations.empty:
                        recommendations = recommendations[recommendations['score_pertinence'] >= min_user_rating]
                        st.write(f"### 🎭 Recommandations similaires à **{best_match}** (excluant le film lui-même) :")
                        st.dataframe(recommendations, use_container_width=True)
                    else:
                        st.warning("Aucune recommandation trouvée après application des filtres.")
            else:
                st.warning("Aucune correspondance trouvée pour votre entrée. Veuillez essayer avec un autre terme.")
        else:
            st.info("Commencez à taper pour voir des recommandations.")

    if choix_principal == "Filtrage Collaboratif : approche modèle":
        sous_choix = st.radio("Choisissez une approche :", ["Surprise (SVD)", "Item-based + SVD"])
        st.write(f"Vous avez sélectionné : {choix_principal} - {sous_choix}")
        
        if sous_choix == "Surprise (SVD)":
            data = df_short[['nom_utilisateur', 'titre_film', 'score_pertinence']]
            reader = Reader(rating_scale=(data['score_pertinence'].min(), data['score_pertinence'].max()))
            dataset = Dataset.load_from_df(data[['nom_utilisateur', 'titre_film', 'score_pertinence']], reader)
            trainset = dataset.build_full_trainset()
            best_params = {'n_factors': 120, 'n_epochs': 20, 'lr_all': 0.003, 'reg_all': 0.85}
            algo = SVD(**best_params)
            algo.fit(trainset)
            
            def get_recommendations_for_user(user_name, algo, data, n=10):
                all_films = data['titre_film'].unique()
                rated_films = data[data['nom_utilisateur'] == user_name]['titre_film'].unique()
                films_to_predict = [film for film in all_films if film not in rated_films]
                predictions = [algo.predict(user_name, film) for film in films_to_predict]
                recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)
                return [(pred.iid, pred.est) for pred in recommendations if pred.iid not in rated_films][:n]
            
            st.subheader("Lancer le Test de Recommandations Personnalisées")
            nom_utilisateur = st.selectbox("Sélectionnez un utilisateur parmi la liste :", data['nom_utilisateur'].unique())
            if st.button("Afficher les recommandations"):
                recommandations = get_recommendations_for_user(nom_utilisateur, algo, data)
                st.write(f"💖 Top 10 des films préférés de {nom_utilisateur} :")
                films_aimes = data[data['nom_utilisateur'] == nom_utilisateur].sort_values(by='score_pertinence', ascending=False).head(10)
                for i, (index, row) in enumerate(films_aimes.iterrows(), 1):
                    etoiles = "⭐" * (int(row['score_pertinence']) // 20)
                    st.markdown(f"{i}. {row['titre_film']}  \nScore donné : {row['score_pertinence']:.2f} {etoiles}")
                st.write(f"🎭 Top 10 des recommandations pour {nom_utilisateur} :")
                for i, (film, score) in enumerate(recommandations, 1):
                    etoiles = "⭐" * (int(score) // 20)
                    st.markdown(f"{i}. {film}  \nScore de pertinence prédit : {score:.2f} {etoiles}")
                st.write("🌟 Top 5 Hors des Sentiers Battus :")
                rated_and_recommended = set(data[data['nom_utilisateur'] == nom_utilisateur]['titre_film'].unique()) | {rec[0] for rec in recommandations}
                random_discovery = data[~data['titre_film'].isin(rated_and_recommended)]['titre_film'].drop_duplicates().sample(5)
                for i, film in enumerate(random_discovery, 1):
                    st.write(f"{i}. {film}")
        

    
        elif sous_choix == "Item-based + SVD":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            from scipy.sparse import csr_matrix
            import pandas as pd
            
            data = df_short[['nom_utilisateur', 'titre_film', 'score_pertinence']]
            pivot_table = data.pivot_table(index='titre_film', columns='nom_utilisateur', values='score_pertinence').fillna(0)
            
            # Convert to sparse matrix
            sparse_ratings = csr_matrix(pivot_table.values)
            
            # Apply Truncated SVD with cross-validation to determine the optimal number of components
            n_components_range = [5, 10, 12, 15, 20, 25, 30]
            best_rmse = float('inf')
            best_n_components = 12
            for n in n_components_range:
                svd = TruncatedSVD(n_components=n)
                ratings_red = svd.fit_transform(sparse_ratings)
                reconstructed_ratings = np.dot(ratings_red, svd.components_)
                rmse = np.sqrt(np.mean((pivot_table.values - reconstructed_ratings) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_n_components = n
            
            # Apply Truncated SVD with the best number of components
            svd = TruncatedSVD(n_components=best_n_components)
            ratings_red = svd.fit_transform(sparse_ratings)
            
            # Calculate item similarity on reduced matrix
            item_similarity = cosine_similarity(ratings_red)
            similarity_df = pd.DataFrame(item_similarity, index=pivot_table.index, columns=pivot_table.index)
            
            reader = Reader(rating_scale=(data['score_pertinence'].min(), data['score_pertinence'].max()))
            dataset = Dataset.load_from_df(data[['nom_utilisateur', 'titre_film', 'score_pertinence']], reader)
            trainset = dataset.build_full_trainset()
            best_params = {'n_factors': 120, 'n_epochs': 20, 'lr_all': 0.003, 'reg_all': 0.85}
            algo = SVD(**best_params)
            algo.fit(trainset)
            
            def get_item_svd_recommendations(user_name, algo, similarity_df, data, n=10):
                user_ratings = data[data['nom_utilisateur'] == user_name]
                rated_films = user_ratings['titre_film'].unique()
                
                # Pre-calculate all similar films once
                recommendations = []
                visited_films = set(rated_films)
                
                for film in rated_films:
                    # Get similar films, sorted by similarity score
                    similar_films = similarity_df[film].sort_values(ascending=False).index
                    for similar_film in similar_films:
                        if similar_film not in visited_films:
                            predicted_rating = algo.predict(user_name, similar_film).est
                            recommendations.append((similar_film, predicted_rating))
                            visited_films.add(similar_film)
                
                # Sort recommendations by predicted rating
                recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
                
                # Remove duplicates and limit to top n recommendations
                seen = set()
                unique_recommendations = []
                for rec in recommendations:
                    if rec[0] not in seen:
                        unique_recommendations.append(rec)
                        seen.add(rec[0])
                    if len(unique_recommendations) == n:
                        break
                
                return unique_recommendations
            
            st.subheader("Lancer le Test de Recommandations Personnalisées")
            nom_utilisateur = st.selectbox("Sélectionnez un utilisateur parmi la liste :", data['nom_utilisateur'].unique())
            if st.button("Afficher les recommandations"):
                recommandations = get_item_svd_recommendations(nom_utilisateur, algo, similarity_df, data)
                st.write(f"💖 Top 10 des films préférés de {nom_utilisateur} :")
                films_aimes = data[data['nom_utilisateur'] == nom_utilisateur].sort_values(by='score_pertinence', ascending=False).head(10)
                for i, (index, row) in enumerate(films_aimes.iterrows(), 1):
                    etoiles = "⭐" * (int(row['score_pertinence']) // 20)
                    st.markdown(f"{i}. {row['titre_film']}  \nScore donné : {row['score_pertinence']:.2f} {etoiles}")
                st.write(f"🎭 Top 10 des recommandations pour {nom_utilisateur} :")
                for i, (film, score) in enumerate(recommandations, 1):
                    etoiles = "⭐" * (int(score) // 20)
                    st.markdown(f"{i}. {film}  \nScore de pertinence prédit : {score:.2f} {etoiles}")
                st.write("🌟 Top 5 Hors des Sentiers Battus :")
                rated_and_recommended = set(data[data['nom_utilisateur'] == nom_utilisateur]['titre_film'].unique()) | {rec[0] for rec in recommandations}
                random_discovery = data[~data['titre_film'].isin(rated_and_recommended)]['titre_film'].drop_duplicates().sample(5)
                for i, film in enumerate(random_discovery, 1):
                    st.write(f"{i}. {film}")

                   



elif option == "Évaluation des Modèles de Recommandation":
    # Évaluation des performances des modèles de recommandation
    st.subheader("Analyse des Performances des Modèles de Recommandation")
    st.write("Dans ce volet, nous allons évaluer les performances des modèles à l'aide de différentes métriques pour mieux comprendre leur efficacité.")
    
    # Ajout d'une clé unique au multiselect
    evaluation_choices = st.multiselect("Choisissez les modèles à évaluer :", [
        "SVD (Surprise) (modele)","Item-based + SVD (modele)", "Filtrage basé sur le contenu", "User-based (approche mémoire)", "Item-based (approche mémoire),"
    ], key="evaluation_choices_key")

    performances = []
    for evaluation_choice in evaluation_choices:
        data = df_short[['nom_utilisateur', 'titre_film', 'score_pertinence']]
        reader = Reader(rating_scale=(data['score_pertinence'].min(), data['score_pertinence'].max()))
        dataset = Dataset.load_from_df(data[['nom_utilisateur', 'titre_film', 'score_pertinence']], reader)
        trainset, testset = train_test_split(dataset, test_size=0.2)

        if evaluation_choice == "SVD (Surprise) (modele)":
            algo = SVD()
            algo.fit(trainset)
            predictions = algo.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            performances.append({"Modèle": "SVD (Surprise)", "RMSE": rmse, "MAE": mae})


        elif evaluation_choice == "Item-based + SVD (modele)":
            try:
                # Convertir les données en un tableau croisé pour les calculs de similarité des items
                pivot_table = data.pivot_table(index='titre_film', columns='nom_utilisateur', values='score_pertinence', aggfunc='mean').fillna(0)

                # Convertir le tableau croisé en une matrice sparse
                sparse_ratings = csr_matrix(pivot_table.values)

                # Appliquer SVD pour réduire la dimensionnalité
                n_components = 10
                svd = TruncatedSVD(n_components=n_components)
                matrice_reduite = svd.fit_transform(sparse_ratings)

                # Calculer la similarité des items sur la matrice réduite
                item_similarity = cosine_similarity(matrice_reduite)
                similarity_df = pd.DataFrame(item_similarity, index=pivot_table.index, columns=pivot_table.index)

                # Fonction de prédiction basée sur la similarité des items
                def predire_notes(testset, similarity_df):
                    predictions = []
                    for uid, iid, true_r in testset:
                        if iid in similarity_df.index:
                            items_similaires = similarity_df.loc[iid].nlargest(10).index  # Les 10 items les plus similaires
                            notes_utilisateur = data[data['nom_utilisateur'] == uid].set_index('titre_film')['score_pertinence']
                            items_notes_par_utilisateur = notes_utilisateur.index.intersection(items_similaires)
                            if not items_notes_par_utilisateur.empty:
                                note_predite = notes_utilisateur.loc[items_notes_par_utilisateur].mean()
                            else:
                                note_predite = pivot_table.loc[iid].mean()  # Revenir à la moyenne des notes de l'item
                        else:
                            note_predite = pivot_table.loc[iid].mean()
                        predictions.append((uid, iid, true_r, note_predite))
                    return predictions

                # Générer les prédictions pour le jeu de test
                predictions = predire_notes(testset, similarity_df)
                rmse = np.sqrt(np.mean([(true_r - est) ** 2 for (_, _, true_r, est) in predictions]))
                mae = np.mean([abs(true_r - est) for (_, _, true_r, est) in predictions])
                performances.append({"Modèle": "Item-based + SVD", "RMSE": rmse, "MAE": mae})

                # Affichage des performances
                if performances:
                    st.subheader("Performances des Modèles de Recommandation")
                    performances_df = pd.DataFrame(performances)
                    st.dataframe(performances_df, use_container_width=True)
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de l'évaluation du modèle 'Item-based + SVD': {e}")



        elif evaluation_choice == "Filtrage basé sur le contenu":
            st.write("Pour le filtrage basé sur le contenu, l'évaluation de la qualité peut se faire via l'analyse de la satisfaction utilisateur, en examinant la pertinence perçue des recommandations et les taux de clics sur les recommandations proposées.")
            st.write("Il est également possible d'utiliser des sondages utilisateurs, des tests A/B, ou de suivre des métriques d'engagement (comme les taux de conversion) pour comprendre l'impact des recommandations sur l'audience.")
            performances.append({"Modèle": "Filtrage basé sur le contenu", "RMSE": "Non applicable", "MAE": "Non applicable"})

        elif evaluation_choice == "User-based (approche mémoire)":
            algo = KNNBasic(k=5, sim_options={'user_based': True})
            algo.fit(trainset)
            predictions = algo.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            performances.append({"Modèle": "User-based (approche mémoire)", "RMSE": rmse, "MAE": mae})

        elif evaluation_choice == "Item-based (approche mémoire)":
            algo = KNNBasic(k=5, sim_options={'user_based': False})
            algo.fit(trainset)
            predictions = algo.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            performances.append({"Modèle": "Item-based (approche mémoire)", "RMSE": rmse, "MAE": mae})

    if performances:
        st.subheader("Tableau Comparatif des Performances des Modèles")
        performance_df = pd.DataFrame(performances)
        st.dataframe(performance_df, use_container_width=True)

elif option == "Évaluation des Modèles de Recommandation Rapide":
    # Évaluation des performances des modèles de recommandation
    st.subheader("Analyse des Performances des Modèles de Recommandation")
    st.write("Dans ce volet, nous allons évaluer les performances des modèles à l'aide de différentes métriques pour mieux comprendre leur efficacité.")
    
    # Ajout d'une clé unique au multiselect
    evaluation_choices = st.multiselect("Choisissez les modèles à évaluer :", [
        "SVD (Surprise)", "Item-based + SVD (modele)" ,  "Filtrage basé sur le contenu", "User-based (approche mémoire)", "Item-based (approche mémoire)"
    ], key="evaluation_choices_key")

    # Création du tableau comparatif des performances des modèles
    performances = {
        "Modèle": ["SVD (Surprise)","Item-based + SVD (modele)", "User-based (approche mémoire)", "Item-based (approche mémoire)", "Filtrage basé sur le contenu"],
        "RMSE": [15.2936, 13.5542 ,14.5321, 13.8114, "Non applicable"],
        "MAE": [12.3635, 10.2443 ,12.0516, 11.1578, "Non applicable"],

    }

    df_performances = pd.DataFrame(performances)

    # Filtrer les modèles sélectionnés par l'utilisateur
    if evaluation_choices:
        df_filtered = df_performances[df_performances['Modèle'].isin(evaluation_choices)]
        st.write("Tableau Comparatif des Performances des Modèles Sélectionnés:")
        st.table(df_filtered)
    else:
        st.write("Veuillez sélectionner au moins un modèle pour voir les résultats.")
