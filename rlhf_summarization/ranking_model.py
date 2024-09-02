# Contains RankingModel class
@dataclass
class RankingModelOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RankingModel(nn.Module):
    def __init__(self, model_name="distilroberta-base"):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        rewards = self.regressor(pooled_output)
        return rewards
