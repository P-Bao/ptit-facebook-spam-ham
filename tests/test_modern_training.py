import json
import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, DataCollatorWithPadding
from pipeline.modern.predict import VibertPredictor
from pipeline.modern.train import ClassificationTrainer, CommentDataset, compute_metrics, make_arguments


def tiny_model(tmp_path):
    vocab=tmp_path/'vocab.txt'
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nxin\nchào\nmua\nhàng\nPTIT\n",encoding='utf-8')
    tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=False)
    model=BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer),hidden_size=16,
        num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,num_labels=2,id2label={0:'ham',1:'spam'},label2id={'ham':0,'spam':1}))
    return model,tokenizer


def test_modern_weighted_loss_and_saved_predictor(tmp_path):
    torch.set_num_threads(1)
    model,tokenizer=tiny_model(tmp_path)
    config={'learning_rate':2e-5,'weight_decay':.01,'warmup_ratio':.1,'batch_size':2,
            'eval_batch_size':2,'gradient_accumulation_steps':1,'seed':42}
    frame=pd.DataFrame({'model_text':['xin chào','mua hàng']*2,'target':[0,1]*2})
    dataset=CommentDataset(frame,tokenizer,16)
    trainer=ClassificationTrainer(model=model,args=make_arguments(tmp_path/'run',config,1,True),
        train_dataset=dataset,eval_dataset=dataset,processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=compute_metrics,
        loss_weights=[1,3])
    batch=trainer.data_collator([dataset[0],dataset[1]])
    loss,output=trainer.compute_loss(model,batch,return_outputs=True)
    expected=torch.nn.functional.cross_entropy(output.logits,batch['labels'],weight=torch.tensor([1.,3.]))
    torch.testing.assert_close(loss,expected)
    trainer.train()
    assert any('eval_f1' in row for row in trainer.state.log_history)
    saved=tmp_path/'model'
    trainer.save_model(saved)
    tokenizer.save_pretrained(saved)
    (saved/'inference_config.json').write_text(json.dumps({'max_length':16,'threshold':.5}),encoding='utf-8')
    predictor=VibertPredictor(saved,device='cpu')
    first=predictor.predict(['xin chào','mua hàng'])
    second=VibertPredictor(saved,device='cpu').predict(['xin chào','mua hàng'])
    np.testing.assert_array_equal(first.probability_spam,second.probability_spam)
    assert first.du_doan.tolist()==second.du_doan.tolist()
    assert predictor.predict([]).empty
