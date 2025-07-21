'''
 * 使用现有Alpha BLIP模型的训练脚本 - 修复版本
 * 直接整合您的alpha_blip_model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoProcessor
import transformers
import copy
transformers.logging.set_verbosity_error()

# 导入您的Alpha BLIP模型
from model.alpha_blip import AlphaBlipForConditionalGeneration, AlphaBlipVisionModel

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0] 
    return tokenizer


class AlphaBlipPretrainModel(nn.Module):
    """
    结合您的Alpha BLIP模型与原始BLIP预训练的三任务训练
    """
    def __init__(self,
                 pretrained_model_name="Salesforce/blip-image-captioning-base",
                 embed_dim=256,
                 queue_size=57600,
                 momentum=0.995,
                ):
        super().__init__()
        
        self.alpha_blip = AlphaBlipForConditionalGeneration.from_pretrained_alpha(
            pretrained_model_name
        )
        
        # 2. 获取模型组件
        self.visual_encoder = self.alpha_blip.vision_model  # 您的Alpha视觉编码器
        self.text_decoder = self.alpha_blip.text_decoder   # 文本解码器
        
        # 3. 创建额外的组件以支持三任务训练
        vision_width = self.visual_encoder.config.hidden_size
        text_width = self.text_decoder.bert.config.hidden_size
        
        # 投影层
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        
        # ITM分类头
        self.itm_head = nn.Linear(text_width, 2)
        
        # 直接使用您的Alpha BLIP模型中的tokenizer和text_encoder
        processor = AutoProcessor.from_pretrained(pretrained_model_name)
        base_tokenizer = processor.tokenizer
        
        base_tokenizer.add_special_tokens({'bos_token':'[DEC]'})
        base_tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
        base_tokenizer.enc_token_id = base_tokenizer.additional_special_tokens_ids[0]
        
        self.tokenizer = base_tokenizer
        self.text_encoder = self.alpha_blip.text_decoder.bert
        
        # 扩展模型的词汇表以包含新的特殊token
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        print(f"✅ 成功添加特殊token:")
        print(f"   [DEC] (bos_token_id): {self.tokenizer.bos_token_id}")
        print(f"   [ENC] (enc_token_id): {self.tokenizer.enc_token_id}")
        print(f"   词汇表大小: {len(self.tokenizer)}")
        
        # 一次性检查特殊token，避免每次前向传播都打印
        self._check_special_tokens()
        
        # 4. 创建动量编码器 - 直接在这里创建，不调用方法
        print("🔧 创建动量编码器...")
        
        # 创建动量视觉编码器
        self.visual_encoder_m = AlphaBlipVisionModel(self.visual_encoder.config)
        self.visual_encoder_m.load_state_dict(self.visual_encoder.state_dict())
        for param in self.visual_encoder_m.parameters():
            param.requires_grad = False
        
        # 创建动量文本编码器
        self.text_encoder_m = copy.deepcopy(self.text_encoder)
        for param in self.text_encoder_m.parameters():
            param.requires_grad = False
            
        # 投影层的动量版本
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        # 模型配对（用于参数复制和动量更新）
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                           [self.vision_proj, self.vision_proj_m],
                           [self.text_encoder, self.text_encoder_m],
                           [self.text_proj, self.text_proj_m]]
        
        # 复制初始参数
        self.copy_params()
        
        # 5. 创建特征队列
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        # 6. 训练参数
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        print("✅ Alpha BLIP预训练模型构建完成")
        
    def _check_special_tokens(self):
        """一次性检查tokenizer的特殊token"""
        self.has_enc_token = hasattr(self.tokenizer, 'enc_token_id')
        self.has_bos_token = (hasattr(self.tokenizer, 'bos_token_id') 
                             and self.tokenizer.bos_token_id is not None)
        
        print(f"🔍 Tokenizer特殊token检查:")
        if self.has_enc_token:
            print(f"   ✅ enc_token_id: {self.tokenizer.enc_token_id}")
        else:
            print(f"   ⚠️  没有enc_token_id，将使用cls_token_id: {self.tokenizer.cls_token_id}")
            
        if self.has_bos_token:
            print(f"   ✅ bos_token_id: {self.tokenizer.bos_token_id}")
        else:
            print(f"   ⚠️  没有bos_token_id，将使用cls_token_id: {self.tokenizer.cls_token_id}")
        
        print(f"   📋 可用特殊token: cls_token_id={self.tokenizer.cls_token_id}")
        
    def _extract_vision_features(self, vision_model, pixel_values, alpha_values=None):
        """统一的视觉特征提取函数"""
        vision_outputs = vision_model(
            pixel_values=pixel_values,
            alpha_values=alpha_values,
            return_dict=True
        )
        
        if hasattr(vision_outputs, 'last_hidden_state'):
            return vision_outputs.last_hidden_state
        else:
            return vision_outputs
        
    def forward(self, image, caption, alpha_values=None, alpha_weight=0.4):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        
        # === 1. Image-Text Alignment (ITA) ===
        
        # 视觉特征提取（支持Alpha通道）
        image_embeds = self._extract_vision_features(
            self.visual_encoder, image, alpha_values
        )
            
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        
        # 文本特征提取
        text = self.tokenizer(caption, padding='max_length', truncation=True, 
                             max_length=30, return_tensors="pt").to(image.device)
        
        text_output = self.text_encoder(text.input_ids, 
                                      attention_mask=text.attention_mask,
                                      return_dict=True)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        
        # 动量特征提取
        with torch.no_grad():
            self._momentum_update()
            
            # 动量视觉编码器（支持Alpha通道）
            image_embeds_m = self._extract_vision_features(
                self.visual_encoder_m, image, alpha_values
            )
                
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            
            text_output_m = self.text_encoder_m(text.input_ids,
                                              attention_mask=text.attention_mask,
                                              return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)
            
            sim_i2t_targets = alpha_weight * F.softmax(sim_i2t_m, dim=1) + (1 - alpha_weight) * sim_targets
            sim_t2i_targets = alpha_weight * F.softmax(sim_t2i_m, dim=1) + (1 - alpha_weight) * sim_targets
        
        # 计算ITA损失
        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
        
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        
        loss_ita = (loss_i2t + loss_t2i) / 2
        
        # 更新队列
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        
        # === 2. Image-Text Matching (ITM) ===
        encoder_input_ids = text.input_ids.clone()
        if self.has_enc_token:
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        else:
            encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
        
        # 正样本
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                     attention_mask=text.attention_mask,
                                     encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=image_atts,
                                     return_dict=True)
        
        # 困难负样本挖掘
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)
        
        # 选择负样本
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
        # 构建负样本输入
        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        
        output_neg = self.text_encoder(text_ids_all,
                                     attention_mask=text_atts_all,
                                     encoder_hidden_states=image_embeds_all,
                                     encoder_attention_mask=image_atts_all,
                                     return_dict=True)
        
        # ITM损失计算
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :],
                                  output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                               torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        
        # === 3. Language Modeling (LM) ===
        decoder_input_ids = text.input_ids.clone()
        if self.has_bos_token:
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        else:
            decoder_input_ids[:, 0] = self.tokenizer.cls_token_id
            
        decoder_targets = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100)
        
        # 使用您的文本解码器
        decoder_output = self.text_decoder(decoder_input_ids,
                                         attention_mask=text.attention_mask,
                                         encoder_hidden_states=image_embeds,
                                         encoder_attention_mask=image_atts,
                                         labels=decoder_targets,
                                         return_dict=True)
        
        loss_lm = decoder_output.loss
        
        return loss_ita, loss_itm, loss_lm
    
    @torch.no_grad()
    def copy_params(self):
        """复制参数到动量编码器"""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """动量更新"""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        """队列更新"""
        # 收集所有GPU的特征（如果使用分布式训练）
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        
        # 更新队列
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr


def train_epoch_with_three_losses(model, dataloader, optimizer, epoch, device):
    """使用三个损失函数的训练循环"""
    model.train()
    total_loss_ita = 0
    total_loss_itm = 0  
    total_loss_lm = 0
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # 准备数据
        images = batch['images'].to(device)
        captions = batch['captions']  # 文本列表
        alpha_values = batch.get('alpha_values')
        if alpha_values is not None:
            alpha_values = alpha_values.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播 - 获取三个损失
        loss_ita, loss_itm, loss_lm = model(
            image=images,
            caption=captions,
            alpha_values=alpha_values,
            alpha_weight=0.4
        )
        
        # 总损失
        loss = loss_ita + loss_itm + loss_lm
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss_ita += loss_ita.item()
        total_loss_itm += loss_itm.item()
        total_loss_lm += loss_lm.item()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}')
            print(f'  ITA Loss: {loss_ita.item():.4f}')
            print(f'  ITM Loss: {loss_itm.item():.4f}')
            print(f'  LM Loss: {loss_lm.item():.4f}')
            print(f'  Total Loss: {loss.item():.4f}')
    
    # 平均损失
    avg_loss_ita = total_loss_ita / num_batches
    avg_loss_itm = total_loss_itm / num_batches
    avg_loss_lm = total_loss_lm / num_batches
    avg_total_loss = total_loss / num_batches
    
    print(f'Epoch {epoch} 完成:')
    print(f'  平均 ITA Loss: {avg_loss_ita:.4f}')
    print(f'  平均 ITM Loss: {avg_loss_itm:.4f}')
    print(f'  平均 LM Loss: {avg_loss_lm:.4f}')
    print(f'  平均总损失: {avg_total_loss:.4f}')
    
    return avg_loss_ita, avg_loss_itm, avg_loss_lm, avg_total_loss


# 辅助函数
@torch.no_grad()
def concat_all_gather(tensor):
    """分布式训练时收集所有GPU的张量"""
    if not torch.distributed.is_initialized():
        return tensor
        
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def create_alpha_blip_pretrain_model(pretrained_model_name="Salesforce/blip-image-captioning-base", **kwargs):
    """创建Alpha BLIP预训练模型"""
    model = AlphaBlipPretrainModel(pretrained_model_name=pretrained_model_name, **kwargs)
    return model


# 测试函数
def test_three_losses():
    """测试三个损失函数"""
    print("🧪 测试Alpha BLIP三损失训练")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model = create_alpha_blip_pretrain_model()
        model = model.to(device)
        
        # 创建测试数据
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        alpha_values = torch.randn(batch_size, 1, 224, 224).to(device)
        captions = ["a photo of a cat", "a dog in the park"]
        
        # 测试前向传播
        model.train()
        loss_ita, loss_itm, loss_lm = model(
            image=images,
            caption=captions,
            alpha_values=alpha_values,
            alpha_weight=0.4
        )
        
        print("✅ 三损失测试成功!")
        print(f"   ITA Loss: {loss_ita.item():.4f}")
        print(f"   ITM Loss: {loss_itm.item():.4f}")
        print(f"   LM Loss: {loss_lm.item():.4f}")
        print(f"   Total Loss: {(loss_ita + loss_itm + loss_lm).item():.4f}")
        
        # 测试反向传播
        total_loss = loss_ita + loss_itm + loss_lm
        total_loss.backward()
        print("✅ 反向传播测试成功!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    if test_three_losses():
        print("\n🎉 Alpha BLIP三损失训练整合成功!")
        print("📋 现在您可以:")
        print("   1. 使用 create_alpha_blip_pretrain_model() 创建模型")
        print("   2. 使用 train_epoch_with_three_losses() 进行训练")
        print("   3. 模型同时支持原始BLIP的三个损失和您的Alpha通道功能")
    else:
        print("\n❌ 整合失败，请检查配置")