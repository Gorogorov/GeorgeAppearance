# Распознавание Святого Георгия

## Описание

Это небольшой проект для практики DL и CV скиллов. Обученные модели определяют, есть ли на изображении Святой Георгий. Я использовал fine-tuning моделей EfficientNet-B2 и ResNet-50. Полученное accuracy 95.8%.

### Установка
Убедитесь, что у вас установлен Python версии не ниже 3.9. Для установки Poetry, следуйте инструкции https://python-poetry.org/docs/#installation

## Запуск
Команды ниже установят необходимые зависимости и откроют ноутбук с описанием проекта.

```
poetry install
poetry shell
jupyter-notebook &
```

## TODO
Прикрутить web: FastAPI + AWS + Docker.

## Датасет
Картинки были найдены на просторах интернета, а затем просмотрены и отфильтрованы глазами. Вы можете скачать датасет [отсюда](https://github.com/Gorogorov/GeorgeAppearance/blob/main/data/georges.csv) и [отсюда](https://github.com/Gorogorov/GeorgeAppearance/blob/main/data/non_georges.csv). Он содержит 5988 изображений, разделенных на 2 категории: 2630 со Святым Георгием и 3358 без него.

## Обучение и предсказание
Инициализация моделей и оптимизаторов, подбор гиперпараметров и краткий анализ результатов содержатся в [ноутбуке](https://github.com/Gorogorov/GeorgeAppearance/blob/main/notebooks/TrainAndResults.ipynb). Обученные модели содержатся в директории [models](https://github.com/Gorogorov/GeorgeAppearance/blob/main/models).

## Результаты
Обе модели получили схожие результаты, однако, исходя из размеров моделей, предпочтительнее использовать EfficinetNet. Краткая сводка результов (больше - в ноутбуке, либо можно посмотреть директорию [md_images](https://github.com/Gorogorov/GeorgeAppearance/blob/main/md_images)):

|Model			    |#Params      |Accuracy		|F1 Score		|Presicion	|Recall		|ROC-AUC
|---------------|---------------|---------------|-----------|-----------|----------|----------|
|EfficientNet-B2		|9.2M			|0.957			|0.949		|0.949		|0.949       |0.987        |
|ResNet-50	        |26M			|0.958			|0.95		  |0.971		|0.93        |0.984        |
