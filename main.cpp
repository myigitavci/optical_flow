// Source codes for EE_576 Project 5
// Mehmet Yiğit Avcı
// Bogazici University, 2022

// necessary declarations
#include <header.h>
#include "header.cpp"



int main(int argc, char *argv[])
{

    string data_type;
    string blank_or_image;
    cout << "Enter which dataset to be used."<<endl;
    cin >> data_type;
    cout << "Enter whether to draw vector field on the image or blank image. For image, write 'image', for blank image, write 'blank'"<<endl;
    cin >> blank_or_image;

    //obtaining file names
    std::string folder = "../576_project5/"+data_type+"/img/*.jpg";
    std::vector<std::string> filenames;
    cv::glob(folder, filenames);
    size_t N = filenames.size() ;

    // obtaining the results
     dense_optical_flow(data_type,blank_or_image,filenames,N);

     sparse_optical_flow(data_type,filenames,N);

     stitch_images(data_type,filenames,N);

     draw_matched_features(data_type,filenames,N);


    return 0;
}
