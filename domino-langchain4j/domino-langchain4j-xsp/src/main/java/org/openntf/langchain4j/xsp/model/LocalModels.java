package org.openntf.langchain4j.xsp.model;

import dev.langchain4j.model.embedding.AbstractInProcessEmbeddingModel;
import dev.langchain4j.model.embedding.OnnxBertBiEncoder;
import dev.langchain4j.model.embedding.PoolingMode;

import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.MessageFormat;
import java.util.HashMap;
import java.util.Map;

/**
 * TODO This should be replaced with a more generic implementation.
 * Experimental implementation to load local ONNX Models. For future implementation:
 * - Complete the implementation of the ONNX model loader
 * - Model maintainer to keep model files distributed and updated [DOTS?]
 * - Model loader to load models from the file system
 */
public class LocalModels {

    private static final Map<String, InProcessEmbeddingModel> modelMap = new HashMap<>();

    /**
     * Load an ONNX model from the local file system.
     * - Set notes.ini parameter - ONNX_REPO_DIR to point out the model repository directory (Default: /local/onnx)
     * - Model files should be in the model repository directory in the following structure:
     * - [ONNX_REPO_DIR]/[model_name]/[model_name].onnx
     * - [ONNX_REPO_DIR]/[model_name]/tokenizer.json
     *
     * @param modelName name of the model, also the name of the directory containing the model files
     * @return the loaded model. If not loaded before, it stores it.
     */
    public static AbstractInProcessEmbeddingModel getOnnxModel(String modelName) {
        if (modelMap.containsKey(modelName)) {
            return modelMap.get(modelName);
        }

        // Else, we will load the model
        return loadOnnxModel(modelName);
    }

    // TODO In sentence-transformer library, you can also download the model from the internet. As long as we can define
    //  good repositories for the models, they can be downloaded from the internet. Also, we can have a model site
    //  database in the future, though, some models might be really large.

    private static InProcessEmbeddingModel loadOnnxModel(String modelName) {
        // Need a better way to handle this. Synchronized block is not a good idea.
        synchronized (modelMap) {
            FileSystem fs = FileSystems.getDefault();
            Path onnxPath = fs.getPath(getRepoDir(), modelName, modelName + ".onnx");
            Path tokenizerPath = fs.getPath(getRepoDir(), modelName, "tokenizer.json");

            if (Files.exists(onnxPath) && Files.exists(tokenizerPath)) {
                InProcessEmbeddingModel model = new InProcessEmbeddingModel(modelName, onnxPath, tokenizerPath, PoolingMode.MEAN);
                modelMap.put(modelName, model);

                System.out.println("Model loaded locally: " + modelName);

                return model;
            } else {
                String errorMsg = MessageFormat.format(
                        "Model for [{0}] and its tokenizer file are not found in the model directory: [{1}], [{2}]",
                        modelName, onnxPath, tokenizerPath
                );
                throw new RuntimeException(errorMsg);
            }
        }
    }


    /**
     * Get the directory where the ONNX models are stored.
     *
     * @return the directory where the ONNX models are stored
     */
    private static String getRepoDir() {
        // TODO: Directory should be configurable from the notes.ini file. Need to implement some domino stuff.
        // String repoDir = DominoUtils.getEnvironmentString("ONNX_REPO_DIR");
        String repoDir = System.getProperty("ONNX_REPO_DIR");

        if (repoDir == null || repoDir.isEmpty()) {
            return "/local/onnx";
        }
        return repoDir;
    }

    static class InProcessEmbeddingModel extends AbstractInProcessEmbeddingModel {

        private final OnnxBertBiEncoder model;

        public InProcessEmbeddingModel(String modelName, Path modelPath, Path tokenizerPath, PoolingMode poolingMode) {
            try {
                this.model = new OnnxBertBiEncoder(
                        Files.newInputStream(modelPath),
                        Files.newInputStream(tokenizerPath),
                        poolingMode
                );
            } catch (IOException e) {
                String errorMsg = MessageFormat.format(
                        "Error opening the model and/or tokenizer files for model {0} "+
                                "from '{1}' with tokenizer '{2}'", modelName, modelPath.toString(), tokenizerPath.toString()
                );
                throw new RuntimeException(errorMsg, e);
            }
        }

        @Override
        protected OnnxBertBiEncoder model() {
            return this.model;
        }
    }


}
