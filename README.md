# Pruning

Идея данной регуляризации заключается в том, чтобы исключить из архитектуры слои, оказывающие наименьшее влияние на конечный результат.    
Тестирование проводилось на основе модифицированной архитектуры ResNet18 (ResNet20), код которой указан в файле `arch.py`. 
Обучение производилось на основе датасета CIFAR10. Результаты работы представлены на графиках ниже (слева - значения по метрике accuracy, справа - ниспадающая функция ошибки)
<p float="left">
  <img src="/accuracy.png" width="450" />
  <img src="/losses.png" width="450" />
</p>
