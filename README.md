# Pruning

## Описание решения

Идея данной регуляризации заключается в том, чтобы исключить из архитектуры слои, оказывающие наименьшее влияние на конечный результат.    
Тестирование проводилось на основе модифицированной архитектуры ResNet18 (ResNet20), код которой указан в файле `arch.py`. 
Обучение производилось на основе датасета CIFAR10. Результаты работы представлены на графиках ниже (слева - значения по метрике accuracy, справа - ниспадающая функция ошибки)
<p float="left">
  <img src="/accuracy.png" width="350" />
  <img src="/losses.png" width="350" />
</p>

**Accuracy** после проведения регуляризации при помощи прунинга:     
train: `95.476`, val: `87.650`;    
**Loss:**     
train: `0.132`, val: `0.430`.    
**Графики:**    
<p float="left">
  <img src="/accuracy_pruned.png" width="350" />
  <img src="/losses_pruned.png" width="350" />
</p>



## Требования к установке
Запуск был произведен под Ubuntu x64 c Python 3.6.9:
1. `torch==1.7.1`
2. `torchvision==0.8.2`
3. `tqdm==4.55.0`
4. `scikit-learn==0.17.2`
5. `seaborn==0.11.0`
6. `numpy==1.19.5`
7. `matplotlib==3.3.3`

