import os,random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from .train import train
from .network import VAE_types
import csv

class api():
    def dva(self,
            adata,
            mode='denoise',
            vae_type='normal',
            hidden_size=(64, 32, 64),
            hidden_dropout=0.,
            batchnorm=True,
            activation='relu',
            init='glorot_uniform',
            network_kwds={},
            epochs=100,  # training args
            reduce_lr=10,
            early_stop=50,
            batch_size=32,
            optimizer='RMSprop',
            learning_rate=0.01,
            random_state=0,
            threads=None,
            verbose=False,
            training_kwds={},
            return_model=False,
            return_info=True,
            copy=False,
            file_path=None
            ):
        # assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
        assert mode in ('denoise', 'latent'), '%s is not a valid mode.' % mode

        # set seed for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        os.environ['PYTHONHASHSEED'] = '0'

        network_kwds = {**network_kwds,
                        'hidden_size': hidden_size,
                        'hidden_dropout': hidden_dropout,
                        'batchnorm': batchnorm,
                        'activation': activation,
                        'init': init,
                        'file_path': file_path,
                        }

        from tensorflow.python.framework.ops import disable_eager_execution
        disable_eager_execution()

        # input_size = output_size = adata.n_vars
        input_size = output_size = len(adata.columns)
        net = VAE_types[vae_type](input_size=input_size,
                                  output_size=output_size,
                                  **network_kwds)
        net.save()
        net.build()


        training_kwds = {**training_kwds,
                         'epochs': epochs,
                         'reduce_lr': reduce_lr,
                         'early_stop': early_stop,
                         'batch_size': batch_size,
                         'optimizer': optimizer,
                         'verbose': verbose,
                         'threads': threads,
                         'learning_rate': learning_rate,
                         'output_dir': file_path,
                         }

        # 划分训练集和测试集
        train_data, test_data = train_test_split(adata, test_size=0.2, random_state=0)
        hist = train(train_data, net, **training_kwds)
        print(f"len(train_data): {len(train_data)}")
        res = net.predict(test_data, mode, return_info, copy)
        adata = res if copy else adata

        if return_info:
            os.makedirs(file_path, exist_ok=True)
            loss_history = os.path.join(file_path, 'loss_history.csv')
            with open(loss_history, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for key, value in hist.history.items():
                    print(f"{key}: {len(value)}")
                    writer.writerow([key, value])

        if return_model:
            return (adata, net) if copy else net
        else:
            return adata if copy else None