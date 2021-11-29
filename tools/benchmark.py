import os
import time
import argparse
import logging
import sys
sys.path.append("../")
sys.path.append("../autotimm")
from configuration import gluon_config_choice
from autotimm.utils.utils import find_best_model, parse_config, write_csv_file, find_best_model_loop, update_kwargs
from autotimm.proxydata.search_proxy_data import ProxyModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model for different kaggle competitions.')
    parser.add_argument('--data_path',
                        type=str,
                        default='',
                        help='train data dir')
    parser.add_argument('--report_path',
                        type=str,
                        default='',
                        help='report dir to save the experiments results')
    parser.add_argument('--dataset',
                        type=str,
                        default='shopee-iet',
                        help='the kaggle competition.')
    parser.add_argument('--output_path',
                        type=str,
                        default='output_path/',
                        help='output path to save results.')
    parser.add_argument('--model_config',
                        type=str,
                        default='default',
                        choices=[
                            'search_models', 'big_models', 'best_quality',
                            'good_quality_fast_inference', 'default_hpo',
                            'default', 'medium_quality_faster_inference'
                        ],
                        help='the model config for autogluon.')
    parser.add_argument('--custom',
                        type=str,
                        default='predict',
                        help='the name of the submission file you set.')
    parser.add_argument(
        '--num_trials',
        type=int,
        default=-1,
        help='number of trials, if negative, use default setting')
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=-1,
        help='number of training epochs, if negative, will use default setting'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=-1,
        help=
        'training batch size per device (CPU/GPU). If negative, will use default setting'
    )
    parser.add_argument('--ngpus-per-trial',
                        type=int,
                        default=1,
                        help='number of gpus to use.')
    parser.add_argument('--train_framework',
                        type=str,
                        default='autogluon',
                        help='train framework')
    parser.add_argument('--task_name', type=str, default='', help='task name')
    parser.add_argument('--load_best_model',
                        type=bool,
                        default=True,
                        help='will load the best model')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='',
                        help='if true, will load the best model and test')
    parser.add_argument('--data_augmention',
                        type=str,
                        default="False",
                        help='Whether use thee data augmention')
    parser.add_argument('--proxy',
                        action='store_true',
                        help='Whether make proxy dataset')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='Whether use thee data augmention')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    if opt.train_framework == "autogluon":
        from autogluon.vision import ImagePredictor
        logger = logging.getLogger('')
        if not opt.checkpoint_path:
            out_dir = os.path.join(opt.output_path, opt.dataset,
                                   opt.model_config)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            tm = time.strftime("%Y%m%d-%H%M", time.localtime())
            tm_dir = os.path.join(out_dir, tm)
            if not os.path.exists(tm_dir):
                os.makedirs(tm_dir)

            output_directory = os.path.join(tm_dir, 'checkpoint/')
            filehandler = logging.FileHandler(
                os.path.join(tm_dir, 'summary.log'))
            streamhandler = logging.StreamHandler()
            logger.setLevel(logging.INFO)
            logger.addHandler(filehandler)
            logger.addHandler(streamhandler)
            logging.info(opt)

        config = gluon_config_choice(opt.dataset,
                                     model_choice=opt.model_config)
        target_hyperparams = config["hyperparameters"]
        tune_hyperparameter = config["hyperparameter_tune_kwargs"]

        train_data_dir = opt.data_path
        val_data_dir = opt.data_path.replace("train", "val")
        test_data_dir = opt.data_path.replace("train", "test")

        train_dataset = ImagePredictor.Dataset.from_folder(train_data_dir)
        val_dataset = ImagePredictor.Dataset.from_folder(val_data_dir)
        test_dataset = ImagePredictor.Dataset.from_folder(test_data_dir)

        if opt.proxy:
            from autotimm.auto.data import TorchImageClassificationDataset
            train_data, val_data, test_data = TorchImageClassificationDataset.from_folders(
                opt.data_path[:-6])
            proxmodel = ProxyModel()
            proxmodel.fit(train_data, val_data)
            saved_path = os.path.join(opt.output_path, opt.dataset)
            proxy_data = proxmodel.generate_proxy_data(train_data=train_data,
                                                       output_dir=saved_path)
            csv_path = os.path.join(saved_path, "proxy_data.csv")
            proxy_data = ImagePredictor.Dataset.from_csv(csv_path)

        if opt.proxy:
            train_data = proxy_data
            tuning_data = None
        else:
            train_data = train_dataset
            tuning_data = val_dataset

        if not opt.checkpoint_path:
            predictor = ImagePredictor(path=output_directory)
            # overwriting default by command line:
            if int(opt.batch_size) > 0:
                target_hyperparams['batch_size'] = int(opt.batch_size)
            if int(opt.num_epochs) > 0:
                target_hyperparams['epochs'] = int(opt.num_epochs)
            if int(opt.num_trials) > 0:
                tune_hyperparameter['num_trials'] = int(opt.num_trials)

            ngpus_per_trial = target_hyperparams.pop('ngpus_per_trial')
            if int(opt.ngpus_per_trial) > 0:
                ngpus_per_trial = int(opt.ngpus_per_trial)

            predictor.fit(train_data=train_data,
                          tuning_data=tuning_data,
                          hyperparameters=target_hyperparams,
                          hyperparameter_tune_kwargs=tune_hyperparameter,
                          ngpus_per_trial=ngpus_per_trial,
                          time_limit=config['time_limit'])

            summary = predictor.fit_summary()
            logger.info('Top-1 val acc: %.3f' % summary['valid_acc'])
            logger.info(summary)

            if opt.proxy:
                logger.info("=" * 10)
                logger.info("Refit the full data by best config")
                logger.info("Update the model config from the searched space")
                target_hyperparams, tune_hyperparameter = update_kwargs(
                    target_hyperparams, tune_hyperparameter,
                    summary['best_config'])

                predictor.fit(train_data=train_dataset,
                              tuning_data=val_dataset,
                              hyperparameters=target_hyperparams,
                              hyperparameter_tune_kwargs=tune_hyperparameter,
                              ngpus_per_trial=ngpus_per_trial,
                              time_limit=config['time_limit'])

            # use the default saved model to evaluate
            val_acc, _ = predictor.evaluate(val_dataset)
            logger.info("*" * 100)
            logger.info(
                'Use the default saved model to evaluate on validation dataset'
            )
            logging.info('Top-1 valid acc: %.5f' % val_acc)

            # use the default saved model to evaluate
            test_acc, _ = predictor.evaluate(test_dataset)
            logger.info("*" * 100)
            logger.info(
                'Use the default saved model to evaluate on Test dataset')
            logging.info('Top-1 test acc: %.5f' % test_acc)

            # load the best checkpoint to evaluate
            if opt.load_best_model:
                predictor = None
                best_checkpoint, best_config, results = find_best_model(
                    checkpoint_dir=output_directory)
                predictor = ImagePredictor().load(best_checkpoint)

                test_acc, _ = predictor.evaluate(test_dataset)
                logger.info("*" * 100)
                logger.info(
                    'Load the best checkpoint to evaluate on Test dataset')
                logging.info('Top-1 test acc: %.5f' % test_acc)

            # save results
            logger.info("*" * 100)
            filename = 'predictor.ag'
            logger.info("Save The final moodel to predictor.ag")
            predictor.save(os.path.join(output_directory, filename))

            fields = [
                "dataset_name", "hpo_type", "train_acc", "valid_acc",
                "test_acc", "model", "batch_size", "learning_rate", "momentum",
                "wd", "data_augmention", "epochs", "search_strategy",
                "input_size", "total_time"
            ]

            model, batch_size, epochs, learning_rate, momentum, wd, input_size = parse_config(
                best_config)

            train_acc = results.get('train_acc')
            valid_acc = results.get('valid_acc')
            total_time = summary.get('total_time')
            search_strategy = tune_hyperparameter.get('searcher', None)

            report = [[
                opt.dataset, opt.model_config, train_acc, valid_acc, test_acc,
                model, batch_size, learning_rate, momentum, wd,
                opt.data_augmention, epochs, search_strategy, input_size,
                total_time
            ]]

            # collect and save the results to csv
            if not os.path.exists(opt.report_path):
                os.makedirs(opt.report_path)
            report_path = os.path.join(opt.report_path, "report.csv")
            logger.info("Write the results to file: %s" % report_path)
            isExists = os.path.exists(report_path)

            if not isExists:
                write_csv_file(report_path, head=fields, data=report)
            else:
                write_csv_file(report_path, head=None, data=report)

            # save the single traing results to csv
            report_path = os.path.join(output_directory, "report.csv")
            logger.info("Write the results to file: %s" % report_path)
            write_csv_file(report_path, head=fields, data=report)
        else:
            predictor = None
            best_checkpoint, best_config, results = find_best_model_loop(
                checkpoint_dir=opt.checkpoint_path)
            predictor = ImagePredictor().load(best_checkpoint)

            test_acc, _ = predictor.evaluate(test_dataset)
            logger.info("*" * 100)
            logger.info('Load the best checkpoint to evaluate on Test dataset')
            logging.info('Top-1 test acc: %.5f' % test_acc)

            model, batch_size, epochs, learning_rate, momentum, wd, input_size = parse_config(
                best_config)

            fields = [
                "dataset_name", "hpo_type", "train_acc", "valid_acc",
                "test_acc", "model", "batch_size", "learning_rate", "momentum",
                "wd", "data_augmention", "epochs", "search_strategy",
                "input_size", "total_time"
            ]

            model, batch_size, epochs, learning_rate, momentum, wd, input_size = parse_config(
                best_config)

            train_acc = results.get('train_acc', None)
            valid_acc = results.get('valid_acc', None)
            total_time = results.get('total_time', None)
            search_strategy = tune_hyperparameter.get('searcher', None)

            report = [[
                opt.dataset, opt.model_config, train_acc, valid_acc, test_acc,
                model, batch_size, learning_rate, momentum, wd,
                opt.data_augmention, epochs, search_strategy, input_size,
                total_time
            ]]

            # collect and save the results to csv
            if not os.path.exists(opt.report_path):
                os.makedirs(opt.report_path)
            report_path = os.path.join(opt.report_path, "report.csv")
            logger.info("Write the results to file: %s" % report_path)
            isExists = os.path.exists(report_path)

            if not isExists:
                write_csv_file(report_path, head=fields, data=report)
            else:
                write_csv_file(report_path, head=None, data=report)

            # save results
            logger.info("*" * 100)
            filename = 'predictor.ag'
            output_directory_list = best_checkpoint.split('/')[:-3]
            output_directory = '/'.join(output_directory_list)
            logger.info("Save The final moodel `predictor.ag` to %s" %
                        output_directory)
            predictor.save(os.path.join(output_directory, filename))

    elif opt.train_framework == "autotimm":
        import torch
        from autotimm.auto import ImagePredictor
        logger = logging.getLogger('')
        if not opt.checkpoint_path:
            out_dir = os.path.join(opt.output_path, opt.dataset,
                                   opt.model_config)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            tm = time.strftime("%Y%m%d-%H%M", time.localtime())
            tm_dir = os.path.join(out_dir, tm)
            if not os.path.exists(tm_dir):
                os.makedirs(tm_dir)

            output_directory = os.path.join(tm_dir, 'checkpoint/')
            filehandler = logging.FileHandler(
                os.path.join(tm_dir, 'summary.log'))
            streamhandler = logging.StreamHandler()
            logger.setLevel(logging.INFO)
            logger.addHandler(filehandler)
            logger.addHandler(streamhandler)
            logging.info(opt)

        config = gluon_config_choice(opt.dataset,
                                     model_choice=opt.model_config)
        target_hyperparams = config["hyperparameters"]
        tune_hyperparameter = config["hyperparameter_tune_kwargs"]

        train_data_dir = opt.data_path
        val_data_dir = opt.data_path.replace("train", "val")
        test_data_dir = opt.data_path.replace("train", "test")

        train_dataset = ImagePredictor.Dataset.from_folder(train_data_dir)
        val_dataset = ImagePredictor.Dataset.from_folder(val_data_dir)
        test_dataset = ImagePredictor.Dataset.from_folder(test_data_dir)

        if opt.proxy:
            from autotimm.auto.data import TorchImageClassificationDataset
            train_data, val_data, test_data = TorchImageClassificationDataset.from_folders(
                opt.data_path[:-6])
            proxmodel = ProxyModel()
            proxmodel.fit(train_data, val_data)
            saved_path = os.path.join(opt.output_path, opt.dataset)
            proxy_data = proxmodel.generate_proxy_data(train_data=train_data,
                                                       output_dir=saved_path)
            csv_path = os.path.join(saved_path, "proxy_data.csv")
            proxy_data = ImagePredictor.Dataset.from_csv(csv_path)

        if opt.proxy:
            train_data = proxy_data
            tuning_data = None
        else:
            train_data = train_dataset
            tuning_data = val_dataset

        if not opt.checkpoint_path:
            predictor = ImagePredictor(log_dir=output_directory)
            # overwriting default by command line:
            if int(opt.batch_size) > 0:
                target_hyperparams['batch_size'] = int(opt.batch_size)
            if int(opt.num_epochs) > 0:
                target_hyperparams['epochs'] = int(opt.num_epochs)
            if int(opt.num_trials) > 0:
                tune_hyperparameter['num_trials'] = int(opt.num_trials)

            ngpus_per_trial = target_hyperparams.pop('ngpus_per_trial')
            if int(opt.ngpus_per_trial) > 0:
                ngpus_per_trial = int(opt.ngpus_per_trial)

            predictor.fit(train_data=train_data,
                          tuning_data=tuning_data,
                          hyperparameters=target_hyperparams,
                          hyperparameter_tune_kwargs=tune_hyperparameter,
                          ngpus_per_trial=ngpus_per_trial,
                          time_limit=config['time_limit'])

            summary = predictor.fit_summary()
            logging.info('Top-1 val acc: %.3f' % summary['valid_acc'])
            logger.info(summary)

            # use the default saved model to evaluate
            val_acc, _ = predictor.evaluate(val_dataset)
            logger.info("*" * 100)
            logger.info(
                'Use the default saved model to evaluate on validation dataset'
            )
            logging.info('Top-1 valid acc: %.5f' % val_acc)

            # use the default saved model to evaluate
            test_acc, _ = predictor.evaluate(test_dataset)
            logger.info("*" * 100)
            logger.info(
                'Use the default saved model to evaluate on Test dataset')
            logging.info('Top-1 test acc: %.5f' % test_acc)

if __name__ == '__main__':
    main()
