from src.etl import run_etl
from src.train import train_model

def run_end_to_end_pipeline():
    print("Starting End-to-End Machine Learning Pipeline \n")
    
    run_etl()
    print("\n" + "="*50 + "\n")
    train_model()
    
    print("\n Full Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_end_to_end_pipeline()