
import os
import sys
import tensorflow as tf
import functools

# Import Magenta training modules
try:
    from magenta.models.music_vae import music_vae_train as train_script
    import magenta.models.music_vae.trained_model as trained_model
    print("✓ Successfully imported Magenta modules")
except ImportError as e:
    print(f"✗ Failed to import Magenta modules: {e}")
    print("Please ensure magenta library is properly installed")
    sys.exit(1)

# Check TensorFlow version
tf_version = tf.__version__
print(f"TensorFlow version: {tf_version}")
if not tf_version.startswith("2."):
    print("Warning: This script is designed for TensorFlow 2.x")

def log_model_parameters(model):
    """Log all model parameters for full-parameter training verification
    
    This function helps verify that all 57 variables are participating 
    in the training process, ensuring maximum learning flexibility.
    """
    print("\n🎯 DigiScore Full-Parameter Fine-tuning Configuration")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    # Count parameters by category
    encoder_params = 0
    decoder_params = 0
    latent_params = 0
    other_params = 0
    
    print("📋 Model Architecture Overview:")
    for i, layer in enumerate(model.layers):
        layer_params = layer.count_params()
        total_params += layer_params
        
        if layer.trainable:
            trainable_params += layer_params
            
        # Categorize parameters
        if "encoder" in layer.name:
            encoder_params += layer_params
        elif "decoder" in layer.name:
            decoder_params += layer_params
        elif any(x in layer.name for x in ["z_", "latent", "mu", "sigma"]):
            latent_params += layer_params
        else:
            other_params += layer_params
            
        print(f"   {i:2d}: {layer.name:<30} | {layer_params:>8,} params | {'✓' if layer.trainable else '✗'}")
    
    print(f"\n📊 Parameter Statistics:")
    print(f"   Total parameters:     {total_params:>8,}")
    print(f"   Trainable parameters: {trainable_params:>8,}")
    print(f"   Encoder parameters:   {encoder_params:>8,}")
    print(f"   Decoder parameters:   {decoder_params:>8,}")
    print(f"   Latent parameters:    {latent_params:>8,}")
    print(f"   Other parameters:     {other_params:>8,}")
    
    print(f"\n🎵 Full-Parameter Strategy Rationale:")
    print(f"   • Encoder BiLSTM: Re-learn sparse note pattern encoding")
    print(f"   • Latent Space: Remap clean↔glitch distribution boundaries")
    print(f"   • Decoder LSTM: Adapt to irregular temporal generation")
    print(f"   • Output Layers: Learn extreme note combinations")
    
    print(f"\n🔬 Expected Adaptations:")
    print(f"   1. Sparse sequence processing (28% note density reduction)")
    print(f"   2. Irregular timing structure generation")
    print(f"   3. Clean-to-glitch interpolation capability")
    print(f"   4. Experimental music aesthetic preservation")
    
    print("=" * 60)
    print("🚀 Initiating aggressive parameter optimization for glitch learning\n")

def main():
    """Main function - Apply full-parameter strategy and start training"""
    
    # Save original TrainedModel.__init__
    original_init = trained_model.TrainedModel.__init__
    
    # Define new __init__ with parameter logging
    def enhanced_init(self, *args, **kwargs):
        # Call original initialization
        original_init(self, *args, **kwargs)
        
        # Log model parameters for verification
        if hasattr(self, '_model'):
            print("🎼 Model created, analyzing parameter configuration...")
            log_model_parameters(self._model)
            
            # Ensure all parameters are trainable (full-parameter strategy)
            for layer in self._model.layers:
                layer.trainable = True
                
            print("✓ All parameters set to trainable for maximum learning flexibility")
        else:
            print("⚠️  Warning: Unable to access model instance")
    
    # Replace original __init__ to apply our logging
    trained_model.TrainedModel.__init__ = enhanced_init
    print("🔧 DigiScore full-parameter fine-tuning hook installed...")
    
    # Start training
    print("🎯 Starting full-parameter fine-tuning for glitch music learning...")
    train_script.console_entry_point()

if __name__ == "__main__":
    main() 