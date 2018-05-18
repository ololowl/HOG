#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;
using Image = Matrix<std::tuple<int, int, int>>;

constexpr double EPS = 1e-5;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

Image BMP_to_colour(BMP *data) { 
    int rows = data->TellHeight();
    int cols = data->TellWidth();
    Image img(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            RGBApixel pix = data->GetPixel(j, i);
            img(i, j) = std::make_tuple(static_cast<int>(pix.Red), static_cast<int>(pix.Green), static_cast<int>(pix.Blue));
        }
    }
    return img;
}

Matrix<int> BMP_to_grey(BMP *data) {
    int rows = data->TellHeight();
    int cols = data->TellWidth();
    Matrix<int> img(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            RGBApixel pix = data->GetPixel(j, i);
            img(i, j) = 0.299 * pix.Red + 0.587 * pix.Green + 0.114 * pix.Blue;
        }
    }
    return img;
}

Matrix<int> mirror_edges(const Matrix<int> &img) {
    uint rad = 1;
    Matrix<int> ans(img.n_rows + 2 * rad, img.n_cols + 2 * rad);

    for (uint i = 0; i < img.n_rows; ++i) {
        for (uint j = 0; j < img.n_cols; ++j) {
            ans(i + rad, j + rad) = img(i, j);
        }
    }
    for (uint i = 0; i < rad; ++i) {
        for (uint j = rad; j < ans.n_cols - rad; ++j) {
            ans(i, j) = img(rad - i - 1, j - rad);
            ans(ans.n_rows - i - 1, j) = img(img.n_rows + i - rad, j - rad);
        }
    }
    for(uint i = rad; i < ans.n_rows - rad; ++i) {
        for (uint j = 0; j < rad; ++j) {
            ans(i, j) = img(i - rad, rad - j - 1);
            ans(i, ans.n_cols - j - 1) = img(i - rad, img.n_cols + j - rad);
        }
    }
    for (uint i = 0; i < rad; ++i) {
        for (uint j = 0; j < rad; ++j) {
            ans(i, j) = ans(i, 2 * rad - j - 1);
            ans(i, ans.n_cols - j - 1) = ans(i, ans.n_cols - 2 * rad + j);
            ans(ans.n_rows - i - 1, j) = 
                        ans(ans.n_rows - i - 1, 2 * rad - j - 1);
            ans(ans.n_rows - i - 1, ans.n_cols - j - 1) = 
                        ans(ans.n_rows - i - 1, ans.n_cols - 2 * rad + j);
        }
    }
    return ans;
}

class SobelHor {
public:
        static const int vert_radius = 1;
        static const int hor_radius = 1;
        int operator ()(const Matrix<int> &src) const {
            // {{0.0, 0.0, 0.0},
            //  {-1.0, 0.0, 1.0},
            //  {0.0, 0.0, 0.0}};
            return src(1, 2) - src(1, 0);
// return src(0, 2) + src(1, 2) * 2 + src(2, 2) - src(0, 0) - 2 * src(1, 0) - src(2, 0);
        }
};

class SobelVer {
public:
        static const int vert_radius = 1;
        static const int hor_radius = 1;
        int operator ()(const Matrix<int> &src) const {
            // {{0.0, 1.0, 0.0},
            //  {0.0, 0.0, 0.0},
            //  {0.0, -1.0, 0.0}};
            return src(0, 1) - src(2, 1);
// return src(2, 0) + src(2, 1) * 2 + src(2, 2) - src(0, 0) - 2 * src(0, 1) - src(0, 2);
        }
};

Matrix<int> square_root_gamma_compr (const Matrix<int> &img) {
    Matrix<int> ans(img.n_rows, img.n_cols);
    for (uint i = 0; i < img.n_rows; ++i) {
        for (uint j = 0; j < img.n_cols; ++j) {
            ans(i, j) = sqrt(img(i, j));
        }
    }
    return ans;
}

void get_grad(const Matrix<int> &gradientHor, const Matrix<int> &gradientVer, 
              Matrix<float> &gradientAbs, Matrix<float> &gradientAngle) {
    for (uint i = 0; i < gradientHor.n_rows; ++i) {
        for (uint j = 0; j < gradientHor.n_cols; ++j) {
            float sqSum = static_cast<float>(gradientHor(i, j) * gradientHor(i, j) +
                                             gradientVer(i, j) * gradientVer(i, j));
            gradientAbs(i, j) = std::sqrt(sqSum);
            gradientAngle(i, j) = M_PI + std::atan2(static_cast<float>(gradientVer(i, j)), static_cast<float>(gradientHor(i, j)));
        }
    }
}

class LBP {
public:
        static const int vert_radius = 1;
        static const int hor_radius = 1;
        int operator ()(const Matrix<int> &img) const {
            int ans = 0;
            Matrix<int> kernel = {{128, 64, 32},
                                  {  1,  0, 16},
                                  {  2,  4,  8}};
            for(int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if ((img(1, 1) <= img(i, j)) && ((i != 1) && (j != 1))) {
                        ans += kernel(i, j);
                    }
                }
            }
            return ans;
        }
};

//local binary patterns
void lbp(Matrix<int> &img, vector<float> &features) {
    Matrix<int> binary = img.unary_map(LBP());
    const uint nCellsHor = 7;
    const uint nCellsVer = 8;
    const uint nHistogramCells = 256;
    const uint nCells = nCellsHor * nCellsVer;
    vector<float> histograms(nCellsHor * nCellsVer * nHistogramCells, 0);
    // process each pixel into histogram
    for (uint i_idx = 0; i_idx < binary.n_rows; ++i_idx) {
        for (uint j_idx = 0; j_idx < binary.n_cols; ++j_idx) {
            // cell coord
            uint i = i_idx * nCellsVer / binary.n_rows;
            uint j = j_idx * nCellsHor / binary.n_cols;
            uint realIdx = (i * nCellsHor + j) * nHistogramCells + binary(i_idx, j_idx);
            ++histograms[realIdx];
        }
    }
    // nomalize
    for (uint i = 0; i < nCells; ++i) {
        uint start = i * nHistogramCells;
        double norm = 0.0;
        for (uint j = start; j < start + nHistogramCells; ++j) {
            norm += histograms[j] * histograms[j];
        }
        norm = std::sqrt(norm);
        // add to hog
        for (uint j = start; j < start + nHistogramCells; ++j) {
            features.push_back(histograms[j] / norm);
        }
    }
}

void colour_features(Image &img, vector<float> &features) {
    const uint nCellsHor = 8;
    const uint nCellsVer = 8;
    Matrix<int> nElemInCell(nCellsVer, nCellsHor);
    Image averageColour(nCellsVer, nCellsHor);
    for (uint i_idx = 0; i_idx < img.n_rows; ++i_idx) {
        for (uint j_idx = 0; j_idx < img.n_cols; ++j_idx) {
            //cell coord
            uint i = i_idx * nCellsVer / img.n_rows;
            uint j = j_idx * nCellsHor / img.n_cols;
            int r, g, b;
            int rr, gg, bb;
            std::tie(r, g, b) = averageColour(i, j);
            std::tie(rr, gg, bb) = img(i_idx, j_idx);
            averageColour(i, j) = std::make_tuple(r + rr, g + gg, b + bb);
            nElemInCell(i, j)++;
        }
    }
    for (uint i = 0; i < nCellsVer; ++i) {
        for (uint j = 0; j < nCellsHor; ++j) {
            int r, g, b;
            std::tie(r, g, b) = averageColour(i, j);
            int sum = nElemInCell(i, j);
            features.push_back(static_cast<float>(r) / sum / 255);
            features.push_back(static_cast<float>(g) / sum / 255);
            features.push_back(static_cast<float>(b) / sum / 255);
        }
    }
}

// qetup]djc134-_#}

// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures& features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        Matrix<int> img = BMP_to_grey(data_set[image_idx].first);
        Matrix<int> gradientHor = img.unary_map(SobelHor());
        Matrix<int> gradientVer = img.unary_map(SobelVer());
        Matrix<float> gradientAbs(gradientHor.n_rows, gradientHor.n_cols);
        Matrix<float> gradientAngle(gradientHor.n_rows, gradientHor.n_cols);
        get_grad(gradientHor, gradientVer, gradientAbs, gradientAngle);

        const uint nCellsHor = 7;
        const uint nCellsVer = 8;
        const uint nCellsInBlock = 3;
        const int nAngleCells = 17;
        const float deltaAngle = M_PI * 2 / (nAngleCells - 1);
        vector<float> hog(nCellsVer * nCellsHor * nAngleCells, 0.0); //matrix but in vector
        // compute histograms of each cell
        for (uint i_idx = 0; i_idx < gradientHor.n_rows; ++i_idx) {
            for (uint j_idx = 0; j_idx < gradientHor.n_cols; ++j_idx) {
                // cell coord
                uint i = i_idx * nCellsVer / gradientHor.n_rows;
                uint j = j_idx * nCellsHor / gradientHor.n_cols;

                float x = gradientAngle(i_idx, j_idx) / deltaAngle;
                float xCeil = std::ceil(x);
                float xFloor = std::floor(x);
                // coord in histogram
                int idxLeft = std::floor(x);
                int idxRight = std::ceil(x);
                // coord in vector
                int realLeft = (i * nCellsHor + j) * nAngleCells + idxLeft;
                int realRight = (i * nCellsHor + j) * nAngleCells + idxRight;
                // add to histogram
                if ((idxRight == 0) || (xCeil - xFloor < EPS)) {
                    hog[realRight] += gradientAbs(i_idx, j_idx);
                } else if (idxLeft == nAngleCells - 1) {
                    hog[realLeft] += gradientAbs(i_idx, j_idx);
                } else {
                    hog[realLeft] += gradientAbs(i_idx, j_idx) * (xCeil - x) / (xCeil - xFloor);
                    hog[realRight] += gradientAbs(i_idx, j_idx) * (x - xFloor) / (xCeil - xFloor);
                }
            }
        }
        features.emplace_back(vector<float>(), data_set[image_idx].second);
        // normalize cells over blocks 3x3 and concatenate
        for (uint i = 0; i < nCellsVer - nCellsInBlock; ++i) {
            for (uint j = 0; j < nCellsHor - nCellsInBlock; ++j) {
                double norm = 0.0;
                // compute norm
                for (uint k = 0; k < nCellsInBlock; ++k) {
                    const uint t_begin = ((i + k) * nCellsHor + j) * nAngleCells; 
                    for (uint t = t_begin; t < t_begin + nCellsInBlock * nAngleCells; ++t) {
                        norm += hog[t] * hog[t];
                    }
                }
                norm = std::sqrt(norm);
                // add to features vector
                for (uint k = 0; k < nCellsInBlock; ++k) {
                    const uint t_begin = ((i + k) * nCellsHor + j) * nAngleCells;                     
                    for (uint t = t_begin; t < t_begin + nCellsInBlock * nAngleCells; ++t) {
                        if (norm > EPS) {
                            features[image_idx].first.push_back(hog[t] / norm);
                        } else {
                            features[image_idx].first.push_back(0.0);
                        }
                    }
                }
            }
        }
        // add histograms of local binary patters
        lbp(img, features[image_idx].first);
        // add colour features
        Image colouredImg = BMP_to_colour(data_set[image_idx].first);
        colour_features(colouredImg, features[image_idx].first);
    }      
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
