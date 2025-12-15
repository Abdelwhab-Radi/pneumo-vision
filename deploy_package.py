#!/usr/bin/env python3
"""
Deployment Package Creator
Packages trained models and necessary files for deployment
"""

import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class DeploymentPackager:
    """Create deployment package from trained models"""
    
    def __init__(self, results_dir: str = "./results", deployment_dir: str = "./deployment"):
        self.results_dir = Path(results_dir)
        self.deployment_dir = Path(deployment_dir)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_deployment_structure(self):
        """Create deployment directory structure"""
        print("\n" + "="*60)
        print("📦 CREATING DEPLOYMENT PACKAGE")
        print("="*60)
        
        # Create main deployment directory
        deploy_path = self.deployment_dir / f"deploy_{self.timestamp}"
        
        # Create subdirectories
        dirs = [
            deploy_path,
            deploy_path / "models",
            deploy_path / "config",
            deploy_path / "docs",
            deploy_path / "api"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✓ Created deployment structure at: {deploy_path}")
        return deploy_path
    
    def copy_models(self, deploy_path: Path) -> Dict[str, str]:
        """Copy trained models to deployment package"""
        print("\n📂 Copying model files...")
        
        models_source = self.results_dir / "models"
        models_dest = deploy_path / "models"
        
        copied_files = {}
        
        # Copy keras model
        keras_files = list(models_source.glob("*.keras"))
        if keras_files:
            for keras_file in keras_files:
                dest = models_dest / keras_file.name
                shutil.copy2(keras_file, dest)
                copied_files['keras'] = str(dest)
                print(f"  ✓ {keras_file.name}")
        
        # Copy TFLite model
        tflite_files = list(models_source.glob("*.tflite"))
        if tflite_files:
            for tflite_file in tflite_files:
                dest = models_dest / tflite_file.name
                shutil.copy2(tflite_file, dest)
                copied_files['tflite'] = str(dest)
                print(f"  ✓ {tflite_file.name}")
        
        # Copy deployment manifest if exists
        manifest_file = models_source / "deployment_manifest.json"
        if manifest_file.exists():
            dest = models_dest / "deployment_manifest.json"
            shutil.copy2(manifest_file, dest)
            copied_files['manifest'] = str(dest)
            print(f"  ✓ deployment_manifest.json")
        
        return copied_files
    
    def copy_configuration(self, deploy_path: Path):
        """Copy configuration files"""
        print("\n⚙️  Copying configuration files...")
        
        config_dest = deploy_path / "config"
        
        # Copy training config
        config_files = list(self.results_dir.glob("*.json"))
        for config_file in config_files:
            dest = config_dest / config_file.name
            shutil.copy2(config_file, dest)
            print(f"  ✓ {config_file.name}")
        
        # Copy config.py if exists
        config_py = Path("config.py")
        if config_py.exists():
            shutil.copy2(config_py, config_dest / "config.py")
            print(f"  ✓ config.py")
    
    def copy_api_files(self, deploy_path: Path):
        """Copy API-related files"""
        print("\n🚀 Copying API files...")
        
        api_dest = deploy_path / "api"
        
        api_files = ["api.py", "requirements.txt", ".env.example"]
        for api_file in api_files:
            source = Path(api_file)
            if source.exists():
                shutil.copy2(source, api_dest / api_file)
                print(f"  ✓ {api_file}")
    
    def copy_documentation(self, deploy_path: Path):
        """Copy documentation files"""
        print("\n📚 Copying documentation...")
        
        docs_dest = deploy_path / "docs"
        
        doc_files = [
            "DEPLOYMENT_GUIDE.md",
            "README.md",
            "API_REFERENCE.md"
        ]
        
        for doc_file in doc_files:
            source = Path(doc_file)
            if source.exists():
                shutil.copy2(source, docs_dest / doc_file)
                print(f"  ✓ {doc_file}")
            
            # Check in docs directory
            source_in_docs = Path("docs") / doc_file
            if source_in_docs.exists():
                shutil.copy2(source_in_docs, docs_dest / doc_file)
                print(f"  ✓ {doc_file}")
    
    def create_deployment_readme(self, deploy_path: Path, model_files: Dict):
        """Create deployment README"""
        print("\n📝 Creating deployment README...")
        
        readme_content = f"""# Pneumonia Detection - Deployment Package

**Created:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Package Contents

### Models
- **Keras Model**: `models/{Path(model_files.get('keras', 'N/A')).name if 'keras' in model_files else 'N/A'}`
  - Use for standard TensorFlow deployment
  - Includes full model architecture
  
- **TFLite Model**: `models/{Path(model_files.get('tflite', 'N/A')).name if 'tflite' in model_files else 'N/A'}`
  - Use for mobile/edge deployment
  - Optimized with quantization

### Configuration
- Training configuration files
- API configuration settings

### API
- FastAPI server implementation
- Requirements file
- Environment configuration template

### Documentation
- Deployment guide
- API reference
- Usage instructions

## Quick Start

### 1. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2. Start API Server

```bash
python api.py
```

The API will be available at http://localhost:8000

### 3. Test API

Open your browser and navigate to:
- http://localhost:8000/docs - Interactive API documentation
- http://localhost:8000/health - Health check endpoint

## Deployment Options

### Option 1: Local Server
```bash
python api.py
```

### Option 2: Docker
```bash
docker build -t pneumonia-api .
docker run -p 8000:8000 pneumonia-api
```

### Option 3: Cloud Deployment
See `docs/DEPLOYMENT_GUIDE.md` for detailed cloud deployment instructions.

## Model Information

Load the deployment manifest for detailed model information:
```python
import json
with open('models/deployment_manifest.json', 'r') as f:
    manifest = json.load(f)
    print(json.dumps(manifest, indent=2))
```

## Support

For detailed documentation, see the `docs/` directory:
- `DEPLOYMENT_GUIDE.md` - Complete deployment instructions
- `API_REFERENCE.md` - API endpoint documentation (if available)
- `README.md` - Project overview

## Files Checklist

- [ ] Keras model (.keras)
- [ ] TFLite model (.tflite)
- [ ] Deployment manifest
- [ ] Configuration files
- [ ] API server code
- [ ] Documentation

---

**Note**: This is an automated deployment package. Verify all files are present before deploying to production.
"""
        
        readme_path = deploy_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"  ✓ README.md")
    
    def create_deployment_summary(self, deploy_path: Path) -> str:
        """Create deployment summary JSON"""
        summary_path = deploy_path / "deployment_summary.json"
        
        # Try to load manifest for metrics
        manifest_path = deploy_path / "models" / "deployment_manifest.json"
        metrics = {}
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                metrics = manifest.get('performance_metrics', {})
        
        summary = {
            "package_created": datetime.datetime.now().isoformat(),
            "package_name": deploy_path.name,
            "contents": {
                "models": list(Path(deploy_path / "models").glob("*.*")),
                "config": list(Path(deploy_path / "config").glob("*.*")),
                "api": list(Path(deploy_path / "api").glob("*.*")),
                "docs": list(Path(deploy_path / "docs").glob("*.*"))
            },
            "performance_metrics": metrics,
            "deployment_ready": True
        }
        
        # Convert Path objects to strings
        for key in summary['contents']:
            summary['contents'][key] = [str(p.name) for p in summary['contents'][key]]
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_path)
    
    def package(self) -> Path:
        """Create complete deployment package"""
        # Create structure
        deploy_path = self.create_deployment_structure()
        
        # Copy files
        model_files = self.copy_models(deploy_path)
        self.copy_configuration(deploy_path)
        self.copy_api_files(deploy_path)
        self.copy_documentation(deploy_path)
        
        # Create documentation
        self.create_deployment_readme(deploy_path, model_files)
        summary_path = self.create_deployment_summary(deploy_path)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ DEPLOYMENT PACKAGE CREATED")
        print("="*60)
        print(f"\nPackage location: {deploy_path.absolute()}")
        print(f"Summary: {summary_path}")
        print("\nNext steps:")
        print("  1. Review the package contents")
        print("  2. Test the API locally")
        print("  3. Deploy to your target environment")
        print("\nFor detailed instructions, see:")
        print(f"  {deploy_path / 'README.md'}")
        
        return deploy_path


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Create deployment package for trained models")
    parser.add_argument('--results-dir', default='./results', help='Results directory')
    parser.add_argument('--deployment-dir', default='./deployment', help='Deployment output directory')
    
    args = parser.parse_args()
    
    packager = DeploymentPackager(
        results_dir=args.results_dir,
        deployment_dir=args.deployment_dir
    )
    
    deploy_path = packager.package()
    
    print(f"\n🎉 Deployment package ready at: {deploy_path.absolute()}")


if __name__ == "__main__":
    main()
