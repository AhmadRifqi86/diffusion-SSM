import deploy.dockerutil as dockerutil
import torch
from models.diffuse import UShapeMambaDiffusion

def build_trtexec_command(args, engine_path):
    cmd = f"trtexec --onnx={args.onnx} --saveEngine={engine_path}"

    # Precision handling
    precision = args.precision.lower()
    if precision == "fp16":
        cmd += " --fp16"
    elif precision == "int8":
        cmd += " --int8"
    elif precision == "int4":
        cmd += " --int4"
    # fp32 is default, so no flag needed

    if args.inputIOFormats:
        cmd += f" --inputIOFormats={args.inputIOFormats}"
    if args.outputIOFormats:
        cmd += f" --outputIOFormats={args.outputIOFormats}"
    if args.workspace:
        cmd += f" --workspace={args.workspace}"
    if args.batch:
        cmd += f" --shapes=input:{args.batch}x{args.input_shape}"
    if args.extra:
        cmd += f" {args.extra}"
    return cmd

def load_model(config, model_path):
    """
    Load a trained PyTorch model
    
    Args:
        model_type: Type of model (simple_cnn, resnet, densenet)
        model_path: Path to the trained model file (.pt)
        num_classes: Number of output classes
        in_channels: Number of input channels
        
    Returns:
        Loaded PyTorch model in evaluation mode
    """
    print(f"Loading model from {model_path}")
    
    # Create model architecture
    #model = get_model(model_type, num_classes, in_channels)
    model = UShapeMambaDiffusion(config)
    
    # Load trained weights
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model

#def convert_onnx()

#def OptimizeModel()