from asyncio.log import logger
import spacy
from spacy.cli.train import train as fit
import os
from logger import custom_logger

def train():
    """
        train spacy model with balanced data
    """
    model_path = os.path.join(os.getcwd(), 'MODEL')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # initialize customer logger
    custom_logger("logfile.tab")

    fit(
        config_path="config.cfg",
        output_path=model_path,
        use_gpu=-1
    )

    # adding entity ruler
    nlp = spacy.load(os.path.join(model_path, "model-best"))

    pattern_email = [{"label": "EMAIL", "pattern":[{"LOWER": {'REGEX' : "\w+\.\w+\@\w+\.com"}}]}]

    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(pattern_email)

if __name__ == "__main__":
    train()