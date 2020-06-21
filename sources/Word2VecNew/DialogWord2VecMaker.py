from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog

from sources.TextPreprocessing import writeStringToFile
from sources.utils import Profiler, getFilenameFromUserSelection

from sources.Word2VecNew.Word2VecCalculator import *
from sources.Word2VecNew.DialogWord2Vec import Ui_Word2VecDialog as DialogWord2Vec # Импорт UI конвертированного в .py
from sources.common.plot.TsneMplForWidget import TsneMplForWidget
from sources.common.plot.PlotMaker import PlotMaker

import matplotlib
matplotlib.use('Qt5Agg')
from sources.Word2VecNew.DimensionReduction import lsa_algorithm, pca_algorithm, tsvd_tsne_algorithms, pca_tsne_algorithm

# # We import seaborn to make nice plots.
# import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
# rc={"lines.linewidth": 2.5})

# Импортируем файл, работающий с алгоритмом word2vec
# Импортируем файл, работающий с моделью word2vec (создание, загрузка, препроцессинг)
# Импортируем файл, работающий с визуализацией с помощью t-SNE

'''
Основной класс алгоритма word2vec
Загрузка интерфейса
Перенаправление методов по созданию и работе с моделью
'''
class DialogWord2VecMaker(QDialog, DialogWord2Vec):
    def __init__(self, input_dir, filename: str, morph, configurations, parent):
        super().__init__()
        DialogWord2Vec.__init__(self)
        self.setupUi(self)
        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        self.setWindowFlags(flags)

        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.parent = parent
        self.input_dir = input_dir       
        self.output_dir = self.configurations.get(
            "output_files_directory", "output_files") + '/Word2Vec/'    
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)  

        self.all_idf_word_keys = []
        self.texts = []
        self.profiler = Profiler()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)        
        
        # Вкладка создание модели
        self.createModelBtn.clicked.connect(self.create_model)
        self.selectAnotherPathBtn.clicked.connect(self.select_another_text)
        self.recreateModelBtn.clicked.connect(self.recreate_model)
        self.recreateModelBtn.setVisible(False) # Скрыть кнопку повторной тренировки модели

        # Вкладка визуализация модели        
        self.searchQueryGBox.setEnabled(False) # Блокируем элементы до выбора модели
        self.plotVLayout.setEnabled(False) # Блокируем элементы до выбора модели
        self.visualizeBtn.setEnabled(False) # Блокируем элементы до выбора модели
        # self.tsnePCABtn.setEnabled(False) # Блокируем элементы до выбора модели
        # self.tsneTSVDBtn.setEnabled(False) # Блокируем элементы до выбора модели
        # self.tsvdBtn.setEnabled(False) # Блокируем элементы до выбора модели
        self.clearBtn.setEnabled(False) # Блокируем элементы до выбора модели
        # self.visualizeGBox.setHidden(True) # Скрыть box с новыми кнопками

        self.visualizeBtn.clicked.connect(self.visualizeWithAlgorithm)
        # self.tsnePCABtn.clicked.connect(self.visualiseTSNEPCA)
        # self.tsneTSVDBtn.clicked.connect(self.visualiseTSNELSA)
        # self.tsvdBtn.clicked.connect(self.visualiseLSA)

        self.selectModelBtn.clicked.connect(self.select_model_file)
        self.showOnPlotBtn.clicked.connect(self.show_word_on_plot)     
        self.searchQueryBtn.clicked.connect(self.search_word)     
        self.searchQueryBtn.setAutoDefault(True)  # click on <Enter>
        self.searchQueryField.returnPressed.connect(self.searchQueryBtn.click)  # click on <Enter>  
        # self.visualizeBtn.clicked.connect(self.visualiseTSNE)
        self.topNField.returnPressed.connect(self.searchQueryBtn.click)  # click on <Enter>  
        self.clearBtn.clicked.connect(self.clear_plots_layout)
        self.filePathField.setText(self.filename)        
        if filename.endswith('.model'):
            self.set_enable_visualisation(filename)

    def set_calc_and_signals(self):
        self.calculator = Word2VecCalculator(self.filename, self.morph,self.configurations)
        self.calculator.signals.Finished.connect(self.on_calculation_finish)
        self.calculator.signals.PrintInfo.connect(self.on_text_log_add)
        self.calculator.signals.Progress.connect(self.on_model_epoch_end)
        self.calculator.signals.ProgressBar.connect(self.on_progress)
        # self.output_dir = self.configurations.get(
        #     "output_files_directory", "output_files") + '/Word2Vec/'
        
        print("Настройки калькулятора и сигналов заданы")

    def on_progress(self, value):
        self.createModelBar.setValue(value)
        self.repaint()

    def on_text_log_add(self, QString):
        self.createLogTextEdit.append(QString + '\n')
        self.repaint()

    def on_model_epoch_end(self, model, epoch):
        root_path = '{0}{1}'.format(self.output_dir, os.path.basename(os.path.splitext(self.filename)[0]))
        model.wv.save_word2vec_format(
            root_path + 'weight_matrix_epoch' + str(epoch) + '.txt')
        self.createLogTextEdit.append("Данные за эпоху " + str(epoch) + " сохранены по адресу output/Word2Vec")
        
    def beforeVisualize(self):
        self.selectModelBtn.setEnabled(False)
        self.visualizeBtn.setEnabled(False)
        # self.tsnePCABtn.setEnabled(False)
        # self.tsnePCABtn.setEnabled(False)
        # self.tsvdBtn.setEnabled(False)

    def afterVisualize(self, nameAlgorithm):
        self.searchQueryGBox.setVisible(True)
        self.selectModelBtn.setEnabled(True)
        self.visualizeBtn.setEnabled(True)
        # self.tsnePCABtn.setEnabled(True)
        # self.tsneTSVDBtn.setEnabled(True)
        # self.tsvdBtn.setEnabled(True)
        self.clearBtn.setEnabled(True)
        self.visualizeLogTextEdit.append('График алгоритма {0} отображен'.format(nameAlgorithm))

    def drawPlot(self, result):
        self.makePlot = PlotMaker(self.plotVLayout, self)
        # Создаем toolbar (перенесено в PlotMaker)
        self.makePlot.add_toolbar(self)
        ax = self.makePlot.ax
        ax.scatter(result[:, 0], result[:, 1])
        ax.grid(True)
        ax.plot()
        words = list(self.calculator.model.wv.vocab)
        for i, word in enumerate(words):
            ax.annotate(word, xy=(result[i, 0], result[i, 1]))

    def visualiseTSNEPCA(self):
        self.beforeVisualize()  
        X = self.calculator.model.wv[self.calculator.model.wv.vocab]
        result = pca_tsne_algorithm(X)
        self.drawPlot(result)                
        self.afterVisualize("t-SNE со снижением размерности методом PCA")

    def visualisePCA(self):
        self.beforeVisualize()  
        X = self.calculator.model.wv[self.calculator.model.wv.vocab]
        result = pca_algorithm(X, 2)
        self.drawPlot(result)                
        self.afterVisualize("PCA")
    
    def visualiseTSNELSA(self):
        self.beforeVisualize()  
        X = self.calculator.model.wv[self.calculator.model.wv.vocab]
        result = tsvd_tsne_algorithms(X)
        self.drawPlot(result)                
        self.afterVisualize("t-SNE со снижением размерности методом truncated SVD")
    
    def visualiseLSA(self):
        self.beforeVisualize()  
        X = self.calculator.model.wv[self.calculator.model.wv.vocab]
        result = lsa_algorithm(X, 2)
        self.drawPlot(result)                
        self.afterVisualize("truncated SVD")
    
    # Очистка области визуализации.
    def clear_plots_layout(self):
        self.makePlot.removePlot()

    def on_calculation_finish(self):
        self.setEnabled(True)

        self.createLogTextEdit.append(
            'Выполнено за ' + self.profiler.stop() + ' с.')
        
        QApplication.restoreOverrideCursor()
        # self.createModelBtn.setEnabled(True)
        self.recreateModelBtn.setVisible(True) # TODO: Показать кнопку повтора расчетов

        self._log_output_data()
        self.set_enable_visualisation(self.modelFile)
        QMessageBox.information(self, "Внимание", "Создание модели завершено")

    def create_model(self):
        self.createModelBtn.setEnabled(False)
        self.set_calc_and_signals()
        self.createLogTextEdit.append("Исходный файл {0}".format(self.filename))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.calculator.min_count = self.wordFrequencyField.value()
        self.calculator.size = self.vectorSizeField.value()
        self.calculator.learn_rate = self.trainingSpeedField.value()
        self.calculator.window = self.windowField.value()
        self.calculator.negative = self.negativeSamplingField.value()
        self.calculator.ns_exponent = self.negativeSamplingExpField.value()
        self.calculator.sg = 0 if self.CBOWRadio.isChecked() else 1
        self.calculator.iter = self.epochNumberField.value()
        self.calculator.only_nouns = self.nounOnlyCheck.isChecked()
        self.setEnabled(False)
        self.profiler.start()
        self.calculator.start()

    def recreate_model(self):
        print("Повторная тренировка модели")
        # self.set_calc_and_signals()
        self.create_model() # TODO: перетренировать модель, а не создать заново

    def select_another_text(self):
        self.filename = getFilenameFromUserSelection("Text file (*.txt)", self.input_dir)
        self.filePathField.setText(self.filename)
        # self.set_calc_and_signals()

    def select_model_file(self):
        print("Выбрать существующую модель из файла")
        modelFile = getFilenameFromUserSelection("MODEL Files (*.model)", self.output_dir + 'Word2Vec')
        if modelFile != None and len(modelFile.split('/')) > 0:
            self.visualizeLogTextEdit.clear()
            self.set_enable_visualisation(modelFile)

    def search_word(self):
        word = self.searchQueryField.text().strip()

        if word == '' or word is None:
            return
        word = self.morph.parse(word)[0].normal_form
        topNParam = self.topNField.value() if self.topNField.value() != '' else 15
        self.visualizeLogTextEdit.append("Приведение слова к нормальной форме: " + word)
        try:
            result = self.calculator.search_word(word.lower(), topNParam)
            self.visualizeLogTextEdit.append("Поиск слова...")
            self._display_results(word, result)
        except:
            error_text = 'Слово {0} не содержится в словаре'.format(word)
            self.visualizeLogTextEdit.append(error_text)

    def _display_results(self, word, results):
        self.visualizeLogTextEdit.append("Слово {0} употребляется со следующими словами:".format(word))
        self.visualizeLogTextEdit.append(';'.join(["({0} - {1})".format(tpl[0], tpl[1]) for tpl in results]))

    def _log_output_data(self):
        root_path = '{0}{1}'.format(self.output_dir, os.path.basename(os.path.splitext(self.filename)[0]))

        def log_data(name, data):
            outpath = '{0}_{1}.txt'.format(root_path, name)
            writeStringToFile(str(data), outpath)

        def yield_vocab(vocab):
            for key in vocab:
                w = vocab[key]
                # Генератор внутри цикла.Код вызывается и не хранит значения в памяти
                yield '{0} -> count: {1}, index: {2}, sample_int: {3}'.format(key, w.count, w.index, w.sample_int)
        
        log_data('vocab', '\n'.join(
            yield_vocab(self.calculator.model.wv.vocab)))
        self.createLogTextEdit.append("Файл со словарем для текста сохранен по адресу output/Word2Vec")
        log_data('index2word', list(
            enumerate(self.calculator.model.wv.index2word)))
        self.createLogTextEdit.append("Список индексов index2word сохранен по адресу output/Word2Vec")
        self.calculator.model.save(root_path + '_output.model')
        self.createLogTextEdit.append("Модель данных сохранена по адресу output/Word2Vec")
        self.calculator.model.wv.save_word2vec_format(
            root_path + 'weight_matrix.txt')
        self.createLogTextEdit.append("Матрица весов weight_matrix сохранена по адресу output/Word2Vec \n\n")
        self.modelFile = root_path + '_output.model'
    
    def set_enable_visualisation(self, modelFile):
        nameStrArray = modelFile.split('/')
        self.selectModelField.setText(nameStrArray[-3] + '/' + nameStrArray[-2] + '/' + nameStrArray[-1])
        self.visualizeLogTextEdit.append('Модель выбрана')
        self.calculator = Word2VecCalculator(modelFile, self.morph, self.configurations)
        self.visualizeBtn.setEnabled(True) # Делаем доступными элементы после выбора модели
        # self.tsnePCABtn.setEnabled(True) # Делаем доступными элементы после выбора модели
        # self.tsneTSVDBtn.setEnabled(True) # Делаем доступными элементы после выбора модели
        # self.tsvdBtn.setEnabled(True) # Делаем доступными элементы после выбора модели
        self.searchQueryGBox.setEnabled(True) # Делаем доступными элементы после выбора модели
        self.plotVLayout.setEnabled(True) # Делаем доступными элементы после выбора модели
