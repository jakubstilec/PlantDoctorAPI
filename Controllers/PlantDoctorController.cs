using Microsoft.AspNetCore.Mvc;
using PlantDoctorServer.ML;
using System.IO;
using System.Threading.Tasks;

namespace PlantDoctorServer.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PlantDoctorController : ControllerBase
    {
        readonly IImageClassifier _imageProcessor;
        public PlantDoctorController(IImageClassifier imageProcessor)
        {
            _imageProcessor = imageProcessor;
        }

        [HttpGet]
        public string Get()
        {
            return "OK";
        }

        [HttpPost]
        public async Task<string> Post()
        {
            var stream = Request.Body;
            var tmpFile = Path.GetTempFileName() + ".jpg";
            using (var fs = new FileStream(tmpFile, FileMode.Create))
                await stream.CopyToAsync(fs);

            var tag = _imageProcessor.Classify(tmpFile);

            System.IO.File.Delete(tmpFile);

            return tag;
        }



    }

}
