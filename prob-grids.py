import numpy as np
from scipy.stats import expon

import matplotlib.pyplot as plt

np.random.seed(0)
_lambda = 5. # lambda arbritrário, quanto maior mais é aceito valores 'distantes'
F = expon(scale=1. / _lambda).pdf # função exponencial para 'pontuar' as distancias

dim = 226 # dimensão arbitrária para verificação do uso de memória
### grupo de grids artificiais para análise do funcionamento do método
a = np.random.rand(dim, dim)
b = np.eye(dim)[::-1] * 1.5
b2 = np.eye(dim) * 50.
c = np.zeros((dim, dim))
d = np.ones((dim, dim)) 
e = np.ones((dim, dim)) / 2.
x, y = np.meshgrid(np.arange(dim), np.arange(dim))
f = np.cos(x * 0.1) * np.cos(y * 0.1)
g = np.sin(x * 0.1) * np.sin(y * 0.1)

models = np.array([[b, b2, c, d, e, f, g]]) # shape = (1, n_models, lats, lons)
models_dists = np.empty_like(models) # malloc para as distancias
num_models = models.shape[1]
mean = np.mean(models, axis=1) # média inicial dos modelos caso haja necessidade de verificação posterior
median = np.median(models, axis=1) # mediana inicial dos modelos caso haja necessidade de verificação posterior

def gen_dists(models):
    '''
    R*

    Calcula a distância ponto a ponto entre cada um dos modelos e todas as suas
    contra-partes, incluindo a si mesmos. Fazendo de forma a otimizar o
    processo faz-se (1, n, lat, lon) - (n, 1, lat, lon) resultando em
    (n, n, lat, lon) que é um tensor simétrico (n, n) em que cada elemento é
    uma matriz (lat, lon).

    # gen_dists = lambda models: abs(models - models.transpose((1, 0, 2, 3)))

    Parameters
    ----------
    models : np.ndarray
        Array contendo os modelos a serem calculadas as distancias absolutas
        entre seus resultados, este deve estar na forma (1, n, lat, lon).

    Returns
    -------
    : np.ndarray
        Retorna as distancias entre os modelos |X_i - X_j|, com i, j = (1, ..., n)
    '''

    return abs(models - models.transpose((1, 0, 2, 3)))


def gen_weights(models):
    '''
    R*

    Calcula-se o 'peso'/probabilidade das distancias serem aceitas/prováveis
    para serem usadas no cálculo da convergência dos modelos, valores muito
    distantes (probabilidade baixa) serão praticamente descartados quandos
    as médias ponderadas forem calculadas.

    # gen_weights = lambda models: F(gen_dists(models))
    
    Parameters
    ----------
    models : np.ndarray
        Array contendo os modelos a serem calculadas as distancias absolutas
        entre seus resultados, este deve estar na forma (1, n, lat, lon).

    Returns
    -------
    : np.ndarray
        Retorna a pontuação das distancias entre os modelos F(|X_i - X_j|),
        com i, j = 1, ..., n
    '''

    return F(gen_dists(models))


def calc_prob(models, weights):
    '''
    R*

    Faz-se o cálculo m_i = sum(m_i * w_j)/sum(w_j), j = (1, ..., n) para
    cada i = (1, ..., n)
    Para otimização do processo faz-se a multiplicação do elementos
    models e weights com shapes (1, n, lat, lon) e (n, n, lat, lon)
    gerando um tensor em que os modelos então na forma:
    [[m1, m2, m3, ...]]
    
    e que será multiplicada pela função de distância:
    [[F(abs(m1 - m1)), F(abs(m1 - m2)), F(abs(m1 - m3)), ...],
     [F(abs(m2 - m1)), F(abs(m2 - m2)), F(abs(m2 - m3)), ...],
     [F(abs(m3 - m1)), F(abs(m3 - m2)), F(abs(m3 - m3)), ...]]
     
    Obtendo:
    [[m1 * F(abs(m1 - m1)), m2 * F(abs(m1 - m2)), m3 * F(abs(m1 - m3)), ...],
     [m1 * F(abs(m2 - m1)), m2 * F(abs(m2 - m2)), m3 * F(abs(m2 - m3)), ...],
     [m1 * F(abs(m3 - m1)), m2 * F(abs(m3 - m2)), m3 * F(abs(m3 - m3)), ...]]
     
    Disso faz-se a soma no axis=1 para obter:
    [m1 * F(abs(m1 - m1)) + m2 * F(abs(m1 - m2)) + m3 * F(abs(m1 - m3)) + ...,
     m1 * F(abs(m2 - m1)) + m2 * F(abs(m2 - m2)) + m3 * F(abs(m2 - m3)) + ...,
     m1 * F(abs(m3 - m1)) + m2 * F(abs(m3 - m2)) + m3 * F(abs(m3 - m3)) + ...]

    E dividindo pela função de pesos somada também no axis=1:
    [m1 * F(abs(m1 - m1)) + m2 * F(abs(m1 - m2)) + m3 * F(abs(m1 - m3)) + ...,
     m1 * F(abs(m2 - m1)) + m2 * F(abs(m2 - m2)) + m3 * F(abs(m2 - m3)) + ...,
     m1 * F(abs(m3 - m1)) + m2 * F(abs(m3 - m2)) + m3 * F(abs(m3 - m3)) + ...] /
    [F(abs(m1 - m1)) + F(abs(m1 - m2)) + F(abs(m1 - m3)) + ...,
     F(abs(m2 - m1)) + F(abs(m2 - m2)) + F(abs(m2 - m3)) + ...,
     F(abs(m3 - m1)) + F(abs(m3 - m2)) + F(abs(m3 - m3)) + ...]

    Obtendo assim as ponderadas para a próxima iteração:
    m1 = (m1 * F(abs(m1 - m1)) + m2 * F(abs(m1 - m2)) + m3 * F(abs(m1 - m3)) + ...) / (F(abs(m1 - m1)) + F(abs(m1 - m2)) + F(abs(m1 - m3)) + ...)
    m2 = (m1 * F(abs(m2 - m1)) + m2 * F(abs(m2 - m2)) + m3 * F(abs(m2 - m3)) + ...) / (F(abs(m2 - m1)) + F(abs(m2 - m2)) + F(abs(m2 - m3)) + ...)
    m3 = (m1 * F(abs(m3 - m1)) + m2 * F(abs(m3 - m2)) + m3 * F(abs(m3 - m3)) + ...) / (F(abs(m3 - m1)) + F(abs(m3 - m2)) + F(abs(m3 - m3)) + ...)
    ...

    E faz a expansão de dimensão para conservar a forma (1, n, lat, lon)

    # calc_prob = lambda models, weights: np.expand_dims((models * weights).sum(axis=1) / weights.sum(axis=1), 0)

    Parameters
    ----------
    models : np.ndarray
        Array contendo os modelos a serem calculadas as distancias absolutas
        entre seus resultados, este deve estar na forma (1, n, lat, lon).
    weights : np.ndarray
        Array contendo os pesos/pontuação dos pontos de cada modelo em relação
        aos outros modelos e à si mesmo, este deve estar na forma (n, n, lat, lon).

    Returns
    -------
    : np.ndarray
        Array contendo os modelos recalculados, baseado nas pontuações dadas a cada
        ponto. O shape de saída tem mesma forma do models.
    
    '''

    return np.expand_dims((models * weights).sum(axis=1) / weights.sum(axis=1), 0)


# calc_res = lambda dists, weights: np.expand_dims((dists**2 * weights).sum(axis=1) / weights.sum(axis=1), 0) # variance weighted | proposta para calcular resíduo

iters = 10
fig, axs = plt.subplots(iters, num_models, sharex=True, sharey=True)
for i in range(iters):
    for j in range(num_models):
        pcm = axs[i, j].imshow(models[0, j], vmin=0., vmax=models[0, j].max(), cmap='cividis')
        fig.colorbar(pcm, ax=axs[i, j])

    print('max from every model', models.max((0, 2, 3)))
    weights = gen_weights(models)
    models = calc_prob(models, weights)
    
# plt.tight_layout()
plt.show()

'''
plt.subplots()
plt.subplot(2, 2, 1)
plt.imshow(mean[0], cmap='cividis')
plt.title('mean(orig)')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(np.median(models, axis=1)[0], cmap='cividis')
plt.title('median(prob)')
plt.colorbar()

##plt.subplot(2, 2, 3)
##plt.imshow(median[0], cmap='cividis')
##plt.title('median(orig)')
##plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(models_dists[0, 0], cmap='cividis')
plt.title('dists 0')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(np.sqrt(models_dists[0, 1]), cmap='cividis')
plt.title('sqrt dists 0')
plt.colorbar()

##plt.subplot(2, 2, 4)
##plt.imshow((np.median(models, axis=1) - median)[0], cmap='cividis')
##plt.title('median(prob) - median(orig)')
##plt.colorbar()
'''
