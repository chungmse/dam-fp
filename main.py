import os
import subprocess
from datetime import datetime

class DataMiningPipeline:
    def __init__(self):
        """
        Initialize the pipeline with required paths and configurations
        """
        self.src_dir = 'src'
        self.log_dir = 'logs'
        self.start_time = datetime.now()
        self._setup_directories()

    def _setup_directories(self):
        """
        Create necessary directories if they don't exist
        """
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_step(self, step_name, success):
        """
        Log the execution status of each step
        """
        status = "SUCCESS" if success else "FAILED"
        timestamp = datetime.now()
        duration = (timestamp - self.start_time).total_seconds()
        
        log_file = os.path.join(self.log_dir, f'pipeline_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log')
        with open(log_file, 'a') as f:
            f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {step_name}: {status} (Duration: {duration:.2f}s)\n")

    def run_step(self, script_name, step_name):
        """
        Run a single step of the pipeline
        """
        print(f"\n{'='*50}")
        print(f"Running {step_name}...")
        print(f"{'='*50}")
        
        script_path = os.path.join(self.src_dir, script_name)
        try:
            subprocess.run(['python3', script_path], check=True)
            self._log_step(step_name, True)
            print(f"\n{step_name} completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            self._log_step(step_name, False)
            print(f"\nError running {step_name}:")
            print(str(e))
            return False

    def verify_data_files(self):
        """
        Verify that required input data file exists
        """
        required_file = 'datasets/BINANCE_BTCUSDT_D1.csv'
        if not os.path.exists(required_file):
            print(f"Error: Required input file {required_file} not found!")
            return False
        return True

    def run_pipeline(self):
        """
        Run the complete data mining pipeline
        """
        # Verify input data
        if not self.verify_data_files():
            return False

        steps = [
            ('data_preprocessing.py', 'Data Preprocessing'),
            ('technical_indicators.py', 'Technical Indicators Generation'),
            ('pattern_generator.py', 'Pattern Generation'),
            ('apriori_algorithm.py', 'Apriori Algorithm'),
            ('visualization.py', 'Visualization and Analysis')
        ]

        for script, step_name in steps:
            if not self.run_step(script, step_name):
                print("\nPipeline stopped due to error!")
                return False

        print("\nPipeline completed successfully!")
        print(f"Total execution time: {(datetime.now() - self.start_time).total_seconds():.2f} seconds")
        return True

    def print_summary(self):
        """
        Print a summary of the results
        """
        try:
            with open('outputs/analysis_summary.txt', 'r') as f:
                print("\nAnalysis Summary:")
                print("="*50)
                print(f.read())
        except FileNotFoundError:
            print("\nAnalysis summary not found. Make sure the pipeline completed successfully.")

def main():
    print("Starting Bitcoin Price Pattern Analysis Pipeline...")
    print("\nChecking environment and dependencies...")
    
    # Check if required directories exist
    for directory in ['datasets', 'outputs', 'logs', 'src']:
        if not os.path.exists(directory):
            print(f"Creating {directory} directory...")
            os.makedirs(directory)
    
    pipeline = DataMiningPipeline()
    
    if pipeline.run_pipeline():
        pipeline.print_summary()
        print("\nAll results have been saved in the 'outputs' directory.")
        print("Log files can be found in the 'logs' directory.")
    else:
        print("\nPipeline execution failed. Check the logs for details.")

if __name__ == "__main__":
    main()