import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import arviz as az
from sklearn.model_selection import train_test_split

# Configurar la semilla para reproducibilidad
numpyro.set_host_device_count(6)  # Usar 6 cores como en tu modelo brms
RANDOM_SEED = 42  # Semilla para reproducibilidad
rng_key = jax.random.PRNGKey(RANDOM_SEED)

# Cargar los datos
# Asumimos que tienes el archivo creditcard.csv en tu directorio de trabajo
df = pd.read_csv('creditcard.csv')

# Separar en train y test similar al código R proporcionado
np.random.seed(RANDOM_SEED)  # Para reproducibilidad
total_rows = df.shape[0]
train_size = 140000

# Crear índices para train (140,000 filas aleatorias)
id_train = np.random.choice(total_rows, size=train_size, replace=False)

# Crear índices para test (el resto de filas)
id_test = np.array([i for i in range(total_rows) if i not in id_train])

# Dividir los datos
train = df.iloc[id_train]
test = df.iloc[id_test]

print(f"Tamaño del conjunto de entrenamiento: {train.shape[0]} filas")
print(f"Tamaño del conjunto de prueba: {test.shape[0]} filas")

def horseshoe_regression(X, y=None, scale=2.0):
    """
    Modelo de regresión logística con prior horseshoe.
    
    Args:
        X: Matriz de características
        y: Vector de etiquetas binarias (1 para fraude, 0 para no fraude)
        scale: Parámetro de escala para el prior horseshoe
    """
    # Dimensiones
    n_samples, n_features = X.shape
    
    # Prior para el intercept basado en la prevalencia (-6.4 ~ logit(0.0017))
    intercept = numpyro.sample("intercept", dist.Normal(-6.4, 2.0))
    
    # Prior horseshoe para los coeficientes
    # 1. Escala global
    tau = numpyro.sample("tau", dist.HalfCauchy(scale))
    
    # 2. Escalas locales
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(n_features)))
    
    # 3. Coeficientes con prior normal escalado por tau * lambdas (horseshoe)
    beta = numpyro.sample(
        "beta", 
        dist.Normal(0, tau * lambdas)
    )
    
    # Cálculo de logits
    logits = intercept + X @ beta
    
    # Verosimilitud
    with numpyro.plate("data", n_samples):
        # Asegurar que y es float32 para evitar problemas con tipos de datos
        y_float = None if y is None else y.astype(jnp.float32)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y_float)
    
    # Para predicciones
    return logits

def train_model(X_train, y_train, rng_key, num_warmup=500, num_samples=1000, num_chains=4, method="mcmc"):
    """
    Entrenar el modelo con el método especificado: MCMC completo o variacional
    
    Args:
        X_train: Datos de entrenamiento X
        y_train: Etiquetas de entrenamiento y
        rng_key: Clave de aleatoriedad
        num_warmup: Número de iteraciones de calentamiento para MCMC
        num_samples: Número de muestras a generar
        num_chains: Número de cadenas para MCMC
        method: Método de inferencia ("mcmc", "svi", o "laplace")
    """
    # Asegurar que y_train es float32
    y_train = y_train.astype(jnp.float32)
    
    if method == "mcmc":
        # Inicializar el kernel NUTS con control de progreso
        nuts_kernel = NUTS(horseshoe_regression, 
                         target_accept_prob=0.8,
                         adapt_step_size=True)
        
        # Inicializar MCMC con progbar
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
            chain_method="parallel",
        )
        
        # Ejecutar MCMC
        rng_key, rng_key_sample = jax.random.split(rng_key)
        mcmc.run(rng_key_sample, X_train, y_train, scale=2.0)
        
        # Obtener las muestras
        samples = mcmc.get_samples()
        return mcmc, samples, rng_key
        
    elif method == "svi":
        # Para inferencia variacional (mucho más rápida)
        from numpyro.infer import SVI, Trace_ELBO, autoguide
        
        # Guía automática (aproximación variacional)
        guide = autoguide.AutoNormal(horseshoe_regression)
        
        # Optimizador
        optimizer = numpyro.optim.Adam(step_size=0.01)
        
        # Configurar SVI
        svi = SVI(
            horseshoe_regression,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )
        
        # Ejecutar SVI
        rng_key, rng_key_svi = jax.random.split(rng_key)
        svi_result = svi.run(
            rng_key_svi, 
            num_steps=2000,
            X=X_train, 
            y=y_train,
            scale=2.0,
            progress_bar=True
        )
        
        # Obtener muestras de la posterior aproximada
        predictive = Predictive(guide, params=svi_result.params, num_samples=1000)
        rng_key, rng_key_samples = jax.random.split(rng_key)
        samples = predictive(rng_key_samples, X=X_train, y=y_train, scale=2.0)
        
        # No hay objeto MCMC real en este caso
        mcmc = None
        return mcmc, samples, rng_key
        
    elif method == "laplace":
        # Aproximación Laplace (experimental, aún más rápida)
        from numpyro.infer import Laplace
        
        # Inicializar Laplace
        laplace = Laplace(horseshoe_regression)
        
        # Ejecutar optimización MAP
        rng_key, rng_key_laplace = jax.random.split(rng_key)
        laplace_result = laplace.run(
            rng_key_laplace,
            X=X_train,
            y=y_train,
            scale=2.0,
            num_steps=2000,
            progress_bar=True
        )
        
        # Obtener muestras de la posterior aproximada
        samples = laplace_result.samples(num_samples)
        
        # No hay objeto MCMC real en este caso
        mcmc = None
        return mcmc, samples, rng_key
    
    else:
        raise ValueError(f"Método de inferencia desconocido: {method}")

def predict(samples, X_test, rng_key):
    """Hacer predicciones usando las muestras de la posterior"""
    rng_key, rng_key_predict = jax.random.split(rng_key)
    
    try:
        # Método 1: Usar la función Predictive de Numpyro
        predictive = Predictive(horseshoe_regression, samples)
        predictions = predictive(rng_key_predict, X_test)
        
        if "obs" in predictions:
            logits = predictions["obs"]
            # Promedio de las predicciones a través de las muestras
            probs = jnp.mean(jax.nn.sigmoid(logits), axis=0)
        else:
            # Si no tenemos 'obs', calculamos manualmente las probabilidades
            raise KeyError("La clave 'obs' no está en las predicciones")
    
    except Exception as e:
        print(f"Error en predicción con Predictive: {e}")
        # Método 2: Cálculo manual de probabilidades (más seguro)
        print("Usando método alternativo para calcular probabilidades...")
        
        if "intercept" in samples and "beta" in samples:
            # Obtener los parámetros promediados
            intercept = jnp.mean(samples["intercept"])
            beta = jnp.mean(samples["beta"], axis=0)
            
            # Calcular los logits y luego las probabilidades
            logits = intercept + jnp.dot(X_test, beta)
            probs = jax.nn.sigmoid(logits)
        else:
            # Si no encontramos los parámetros esperados, buscar alternativas
            print("Buscando parámetros alternativos en las muestras...")
            param_keys = list(samples.keys())
            print(f"Parámetros disponibles: {param_keys}")
            
            # Buscar los parámetros por convención de nombres
            intercept_key = next((k for k in param_keys if 'intercept' in k.lower()), None)
            beta_key = next((k for k in param_keys if 'beta' in k.lower() or 'weight' in k.lower()), None)
            
            if intercept_key and beta_key:
                intercept = jnp.mean(samples[intercept_key])
                beta = jnp.mean(samples[beta_key], axis=0)
                logits = intercept + jnp.dot(X_test, beta)
                probs = jax.nn.sigmoid(logits)
            else:
                raise ValueError("No se pueden encontrar los parámetros necesarios en las muestras")
    
    return probs, rng_key

# Preparación de datos
def prepare_data(df):
    """Preparar los datos para el modelo"""
    # Seleccionar las mismas columnas que usaste en brms
    features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                'V11', 'V12', 'V13', 'V14', 'Amount']
    X = df[features].values
    y = df['Class'].values
    return X, y

# Entrenamiento y evaluación
def train_and_evaluate(train_df, test_df, rng_key, method="mcmc"):
    """Entrenar el modelo y evaluarlo en el conjunto de prueba"""
    # Preparar los datos
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    
    # Normalizar los datos para mejor convergencia del modelo
    # Especialmente importante para la columna Amount que tiene una escala diferente
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir a arrays de JAX y asegurar tipo float32
    X_train = jnp.array(X_train_scaled, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    X_test = jnp.array(X_test_scaled, dtype=jnp.float32)
    
    # Para conjuntos grandes, podemos usar una muestra para entrenar más rápido
    # Solo si el usuario ha especificado métodos rápidos, usamos el conjunto completo para MCMC
    if method != "mcmc" and X_train.shape[0] > 20000:
        print(f"Usando una muestra de 20,000 filas para entrenamiento más rápido con método {method}")
        # Crear submuestra para entrenamiento más rápido
        subsample_size = min(20000, X_train.shape[0])
        rng_key, subsample_key = jax.random.split(rng_key)
        subsample_idx = jax.random.choice(
            subsample_key, 
            jnp.arange(X_train.shape[0]), 
            shape=(subsample_size,), 
            replace=False
        )
        X_train_sub = X_train[subsample_idx]
        y_train_sub = y_train[subsample_idx]
    else:
        X_train_sub = X_train
        y_train_sub = y_train
    
    # Entrenar el modelo
    print(f"Entrenando modelo con método {method}...")
    print(f"Forma de datos de entrenamiento: {X_train_sub.shape}")
    
    # Medición del tiempo de entrenamiento
    import time
    start_time = time.time()
    
    mcmc, samples, rng_key = train_model(X_train_sub, y_train_sub, rng_key, method=method)
    
    end_time = time.time()
    print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")
    
    # Diagnósticos de convergencia (solo para MCMC)
    if mcmc is not None:
        try:
            summary = az.summary(az.from_numpyro(mcmc))
            print("Resumen de la posterior:")
            print(summary)
        except:
            print("No se pudieron generar diagnósticos de convergencia para este método")
    
    # Hacer predicciones
    print("Haciendo predicciones...")
    probs, rng_key = predict(samples, X_test, rng_key)
    
    # Evaluación
    print("Evaluando el modelo...")
    auc_score = roc_auc_score(test_df['Class'], probs)
    precision, recall, _ = precision_recall_curve(test_df['Class'], probs)
    pr_auc = auc(recall, precision)
    
    print(f"ROC AUC: {auc_score:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    return mcmc, samples, probs

# Guardar los resultados
def save_results(samples, probs, test_df, filename="numpyro_results"):
    """Guardar resultados del modelo"""
    # Guardar las muestras de la posterior
    np.savez(
        f"{filename}.npz",
        **{k: v for k, v in samples.items()}
    )
    
    # Guardar predicciones
    results = pd.DataFrame({
        'true_label': test_df['Class'],
        'predicted_prob': probs
    })
    results.to_csv(f"{filename}_predictions.csv", index=False)
    
    print(f"Resultados guardados en {filename}")

# Método de inferencia: "mcmc" (completo pero lento), "svi" (variacional, más rápido), "laplace" (más rápido aún)
inference_method = "svi"  # Puedes cambiar a "mcmc" para un muestreo completo o "laplace" para una aproximación más rápida
print(f"Usando método de inferencia: {inference_method}")

# Ejecutar el entrenamiento y evaluación
mcmc, samples, probs = train_and_evaluate(train, test, rng_key, method=inference_method)

# Guardar los resultados
save_results(samples, probs, test)
