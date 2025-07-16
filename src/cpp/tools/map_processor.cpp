#include "../maps/spatial_map.h"
#include <iostream>
#include <string>
#include <cstdlib>

using namespace ptm;

void printUsage(const char* progName) {
    std::cout << "Europa Map Processor Utility" << std::endl;
    std::cout << "Usage: " << progName << " <command> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  resample <input.fits> <output.fits> <width> <height> [--global]" << std::endl;
    std::cout << "      Resample a FITS map to a new resolution" << std::endl;
    std::cout << "      Add --global to force global coordinates (0-360°, -90-90°)" << std::endl;
    std::cout << "  info <input.fits> [--global]" << std::endl;
    std::cout << "      Display information about a FITS map" << std::endl;
    std::cout << "      Add --global to force global coordinates (0-360°, -90-90°)" << std::endl;
    std::cout << "  csv <input.fits> <output.csv>" << std::endl;
    std::cout << "      Convert a FITS map to CSV format" << std::endl;
}

bool hasGlobalFlag(int argc, char** argv, int start_index) {
    for (int i = start_index; i < argc; i++) {
        if (std::string(argv[i]) == "--global") {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string command = argv[1];
    
    try {
        if (command == "resample") {
            if (argc < 6) {
                std::cerr << "Error: resample command requires at least 4 arguments" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            
            std::string inputFile = argv[2];
            std::string outputFile = argv[3];
            int width = std::stoi(argv[4]);
            int height = std::stoi(argv[5]);
            
            bool forceGlobal = hasGlobalFlag(argc, argv, 6);
            
            std::cout << "Loading map from " << inputFile << std::endl;
            SpatialMap inputMap;
            if (!inputMap.loadFromFITS(inputFile)) {
                std::cerr << "Error loading map: " << inputMap.getLastError() << std::endl;
                return 1;
            }
            
            if (forceGlobal) {
                std::cout << "Forcing global coordinate mapping (0-360°, -90-90°)" << std::endl;
                inputMap.setCoordinates(0.0, 360.0, -90.0, 90.0);
            }
            
            std::cout << "Resampling to " << width << "x" << height << std::endl;
            std::shared_ptr<SpatialMap> resampledMap = inputMap.resample(width, height);
            
            std::cout << "Saving to " << outputFile << std::endl;
            if (!resampledMap->saveToFITS(outputFile)) {
                std::cerr << "Error saving map: " << resampledMap->getLastError() << std::endl;
                return 1;
            }
            
            std::cout << "Resampling complete" << std::endl;
        }
        else if (command == "info") {
            if (argc < 3) {
                std::cerr << "Error: info command requires at least 1 argument" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            
            std::string inputFile = argv[2];
            
            bool forceGlobal = hasGlobalFlag(argc, argv, 3);
            
            SpatialMap map;
            if (!map.loadFromFITS(inputFile)) {
                std::cerr << "Error loading map: " << map.getLastError() << std::endl;
                return 1;
            }
            
            if (forceGlobal) {
                std::cout << "Forcing global coordinate mapping (0-360°, -90-90°)" << std::endl;
                map.setCoordinates(0.0, 360.0, -90.0, 90.0);
            }
            
            std::cout << "Map information for " << inputFile << ":" << std::endl;
            std::cout << "  Dimensions: " << map.getWidth() << "x" << map.getHeight() << " pixels" << std::endl;
            std::cout << "  Longitude range: " << map.getLonMin() << "° to " << map.getLonMax() << "°" << std::endl;
            std::cout << "  Latitude range: " << map.getLatMin() << "° to " << map.getLatMax() << "°" << std::endl;
            std::cout << "  Value range: " << map.getMin() << " to " << map.getMax() << std::endl;
            std::cout << "  Mean value: " << map.getMean() << std::endl;
            std::cout << "  Standard deviation: " << map.getStdDev() << std::endl;
            std::cout << "  Missing value: " << map.getMissingValue() << std::endl;
        }
        else if (command == "csv") {
            if (argc != 4) {
                std::cerr << "Error: csv command requires 2 arguments" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            
            std::string inputFile = argv[2];
            std::string outputFile = argv[3];
            
            SpatialMap map;
            if (!map.loadFromFITS(inputFile)) {
                std::cerr << "Error loading map: " << map.getLastError() << std::endl;
                return 1;
            }
            
            std::cout << "Converting " << inputFile << " to CSV format" << std::endl;
            if (!map.saveToCSV(outputFile)) {
                std::cerr << "Error saving CSV: " << map.getLastError() << std::endl;
                return 1;
            }
            
            std::cout << "Conversion complete" << std::endl;
        }
        else {
            std::cerr << "Unknown command: " << command << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}