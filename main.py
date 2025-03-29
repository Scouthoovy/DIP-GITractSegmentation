config = {
    'base_dir': r'', 
    'output_dir': r'', 
    'csv_path': r'', 
    'case_id': 'case101',  # Replace with the case ID you want to process
    'height': 266,
    'width': 266,
    'depth': 266,
    'target_height': 128,  
    'target_width': 128,   
    'target_depth': 128,    
    'run_preprocessing': True,
    'run_atlas_creation': True,
    'run_validation': True
}

run_pipeline(config)