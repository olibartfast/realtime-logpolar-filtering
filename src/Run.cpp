#include "rtlp/video/VideoProcessor.h"
#include "rtlp/benchmark/Benchmark.h"
#include <iostream>
#include <string>
#include <cstring>

using namespace std;
using namespace rtlp;

void print_help() {
    cout << "Usage: ./rtlp [MODE] [FILTER] [OPTIONS]" << endl << endl;
    cout << "MODES:" << endl;
    cout << "  --realtime    Process video in real-time" << endl;
    cout << "  --benchmark   Run benchmark tests" << endl << endl;
    cout << "FILTERS (only for --realtime mode):" << endl;
    cout << "  --bilinear         LogPolar direct (Bilinear)" << endl;
    cout << "  --bilinear-inv     LogPolar direct+inverse (Bilinear)" << endl;
    cout << "  --bilinear-gpu     LogPolar direct (Bilinear GPU)" << endl;
    cout << "  --bilinear-gpu-inv LogPolar direct+inverse (Bilinear GPU)" << endl;
    cout << "  --wilson           LogPolar direct (Wilson)" << endl;
    cout << "  --wilson-inv       LogPolar direct+inverse (Wilson)" << endl;
    cout << "  --wilson-gpu       LogPolar direct (Wilson GPU)" << endl;
    cout << "  --wilson-gpu-inv   LogPolar direct+inverse (Wilson GPU)" << endl;
    cout << "  --no-filter        Show original image (no processing)" << endl << endl;
    cout << "BENCHMARK OPTIONS:" << endl;
    cout << "  --image <path>     Image file path (default: test.jpg)" << endl;
    cout << "  --iterations <n>   Number of benchmark iterations (default: 10)" << endl << endl;
    cout << "Examples:" << endl;
    cout << "  ./rtlp --realtime --bilinear" << endl;
    cout << "  ./rtlp --benchmark" << endl;
    cout << "  ./rtlp --benchmark --image myimage.jpg --iterations 50" << endl;
    cout << "  ./rtlp --help" << endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_help();
        return 1;
    }

    rtlp::core::Image img;
    
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_help();
        return 0;
    }
    else if (strcmp(argv[1], "--realtime") == 0) {
        rtlp::video::FilterMode filter = rtlp::video::FilterMode::NONE;
        
        if (argc < 3) {
            cout << "Error: --realtime mode requires a filter argument." << endl;
            print_help();
            return 1;
        }
        
        // Parse filter argument
        if (strcmp(argv[2], "--bilinear") == 0) {
            filter = rtlp::video::FilterMode::BILINEAR;
        }
        else if (strcmp(argv[2], "--bilinear-inv") == 0) {
            filter = rtlp::video::FilterMode::BILINEAR_INV;
        }
        else if (strcmp(argv[2], "--bilinear-gpu") == 0) {
            filter = rtlp::video::FilterMode::BILINEAR_GPU;
        }
        else if (strcmp(argv[2], "--bilinear-gpu-inv") == 0) {
            filter = rtlp::video::FilterMode::BILINEAR_GPU_INV;
        }
        else if (strcmp(argv[2], "--wilson") == 0) {
            filter = rtlp::video::FilterMode::WILSON;
        }
        else if (strcmp(argv[2], "--wilson-inv") == 0) {
            filter = rtlp::video::FilterMode::WILSON_INV;
        }
        else if (strcmp(argv[2], "--wilson-gpu") == 0) {
            filter = rtlp::video::FilterMode::WILSON_GPU;
        }
        else if (strcmp(argv[2], "--wilson-gpu-inv") == 0) {
            filter = rtlp::video::FilterMode::WILSON_GPU_INV;
        }
        else if (strcmp(argv[2], "--no-filter") == 0) {
            filter = rtlp::video::FilterMode::NONE;
        }
        else {
            cout << "Error: Unknown filter '" << argv[2] << "'" << endl;
            print_help();
            return 1;
        }
        
        rtlp::video::VideoProcessor processor;
        processor.SetImage(&img);
        processor.SetFilter(filter);
        processor.show();
    }
    else if (strcmp(argv[1], "--benchmark") == 0) {
        string image_path = "test.jpg";
        int iterations = 10;
        
        // Parse optional benchmark arguments
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
                image_path = argv[i + 1];
                i++; // Skip the next argument as it's the image path
            }
            else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
                iterations = atoi(argv[i + 1]);
                if (iterations <= 0) {
                    cout << "Error: iterations must be a positive number" << endl;
                    return 1;
                }
                i++; // Skip the next argument as it's the iteration count
            }
        }
        
        rtlp::benchmark::Benchmark benchmark(&img, image_path, iterations);
        benchmark.ReadImg();
        benchmark.SaveImg();
        benchmark.Run();
    }
    else {
        cout << "Error: Unknown mode '" << argv[1] << "'" << endl;
        print_help();
        return 1;
    }

    return 0;
}



