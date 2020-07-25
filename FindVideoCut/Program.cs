using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using OpenCvSharp;
namespace FindVideoCut
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("请选择模式，索引：c，查找：s");
            string mode = Console.ReadLine();
            if (!(mode == "c" || mode == "s"))
            {
                Console.WriteLine("未知命令");
                return;
            }
            Console.WriteLine("请输入缓存文件目录");
            string cache = Console.ReadLine().Replace("\"", " ").Trim();
            cache = Path.Combine(cache, "findVideo.cache");
            Cache ch = new Cache();
            bool useChche = false;
            if (File.Exists(cache) && mode == "s")
            {
                try
                {

                    ch = (Cache)DeserializeWithBinary(File.ReadAllBytes(cache));
                    useChche = true;
                }
                catch (Exception)
                {
                    Console.WriteLine("读取缓存错误");
                }
            }

            string pic = "";
            string tarDir = "";
            if (args.Length != 2)
            {
                if (mode == "s")
                {
                    Console.WriteLine("请输入想找的视频的截屏的路径：");
                    pic = Console.ReadLine().Replace("\"", " ").Trim();
                }
                if (mode == "c")
                {
                    Console.WriteLine("请存放视频的文件夹路径：");
                    tarDir = Console.ReadLine().Replace("\"", " ").Trim();
                }


            }
            else
            {
                pic = args[0];
                tarDir = args[1];
            }
            var size = new Size(32, 32);
            var sources = new List<Mat>();
            var source_file = new List<string>();
            if (mode == "s")
            {
                if (Directory.Exists(pic))
                {
                    // 是目录
                    foreach (var item in Directory.GetFiles(pic))
                    {
                        source_file.Add(item);
                        var s = new Mat(item, ImreadModes.Color);
                        var nmt = new Mat();
                        Cv2.Resize(s, nmt, size);
                        sources.Add(nmt);
                    }
                }
                else if (File.Exists(pic))
                {
                    // 是文件
                    var source = new Mat(pic, ImreadModes.Color);
                    var cmpMat = new Mat();
                    Cv2.Resize(source, cmpMat, size);
                    sources.Add(cmpMat);
                }
            }




            double[] maxD = new double[sources.Count];
            double[] sim_per = new double[sources.Count];
            string[] sim_path = new string[sources.Count];
            string[] sim_time = new string[sources.Count];

            if (mode == "s" && useChche)
            {
                foreach (var item in ch.files)
                {
                    for (int i = 0; i < item.smallMat.Count; i++)
                    {
                        var small = item.smallMat[i];
                        for (int j = 0; j < sources.Count; j++)
                        {
                            var cmpMat = sources[j];
                            var d = getPSNR(cmpMat, Mat.FromStream(new MemoryStream(small), ImreadModes.Color));
                            if (d > maxD[j])
                            {
                                maxD[j] = d;
                                double maxI = i;
                                sim_path[j] = item.path;
                                sim_per[j] = maxI / item.smallMat.Count * 100;
                                sim_time[j] = formatLongToTimeStr(maxI * 5 * 1 / item.fps * 1000);
                            }
                        }
                    }
                }
                for (int i = 0; i < source_file.Count; i++)
                {
                    Console.WriteLine("--------------------------");
                    Console.WriteLine($"截屏文件：{source_file[i]}");
                    Console.WriteLine($"对应视频文件：{sim_path[i]}");
                    Console.WriteLine($"相似程度（越大越好）：{maxD[i]}");
                    Console.WriteLine($"最相似：{sim_per[i]}%，大约在{sim_time[i]}");
                }
            }
            else if (mode == "c")
            {

                foreach (var item in Directory.GetFiles(tarDir))
                {
                    var file = new MatFile();
                    file.path = item;
                    try
                    {
                        var video = new VideoCapture(item);
                        var mat = new Mat();
                        var index = 0;
                        double frameount = video.Get(VideoCaptureProperties.FrameCount);
                        double lastp = 0;
                        Console.WriteLine($"当前文件：{item}");
                        while (video.Read(mat))
                        {
                            var per = index / frameount * 100;
                            if (per - lastp > 10)
                            {
                                Console.WriteLine($"...{per} %");
                                lastp = per;
                            }
                            var small = new Mat();
                            Cv2.Resize(mat, small, size);
                            file.smallMat.Add(small.ToMemoryStream().ToArray());

                            video.Grab();
                            video.Grab();
                            video.Grab();
                            video.Grab();
                            index += 5;
                        }
                        file.fps = video.Fps;
                        ch.files.Add(file);

                    }
                    catch (Exception e)
                    {

                    }
                }
                try
                {
                    File.WriteAllBytes(cache, SerializeToBinary(ch));
                }
                catch (Exception)
                {
                    Console.WriteLine("写入缓存失败");
                }
            }
            else
            {
                Console.WriteLine("搜索时请先建立索引。");
            }


        }
        public static string formatLongToTimeStr(double l)
        {
            int hour = 0;
            int minute = 0;
            int second = (int)l / 1000;

            if (second > 60)
            {
                minute = second / 60;
                second %= 60;
            }
            if (minute > 60)
            {
                hour = minute / 60;
                minute %= 60;
            }
            return (hour.ToString() + "小时" + minute.ToString() + "分钟"
                + second.ToString() + "秒");
        }

        static double getPSNR(Mat I1, Mat I2)
        {
            Mat s1 = new Mat();
            Cv2.Absdiff(I1, I2, s1);       // |I1 - I2|
            s1.ConvertTo(s1, MatType.CV_32F);  // cannot make a square on 8 bits
            s1 = s1.Mul(s1);           // |I1 - I2|^2

            Scalar s = Cv2.Sum(s1);         // sum elements per channel

            double sse = s.Val0 + s.Val1 + s.Val2; // sum channels

            if (sse <= 1e-10) // for small values return zero
                return 0;
            else
            {
                double mse = sse / (I1.Channels() * I1.Total());
                double psnr = 10.0 * Math.Log10((255 * 255) / mse);
                return psnr;
            }
        }

        [Serializable]
        class MatFile
        {
            public double fps;
            public string path;
            public List<byte[]> smallMat = new List<byte[]>();
        }

        [Serializable]
        class Cache
        {
            public List<MatFile> files = new List<MatFile>();
        }


        /// <summary>
        /// 将对象序列化为二进制数据 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        static byte[] SerializeToBinary(object obj)
        {
            MemoryStream stream = new MemoryStream();
            BinaryFormatter bf = new BinaryFormatter();
            bf.Serialize(stream, obj);

            byte[] data = stream.ToArray();
            stream.Close();

            return data;
        }


        /// <summary>
        /// 将二进制数据反序列化
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static object DeserializeWithBinary(byte[] data)
        {
            MemoryStream stream = new MemoryStream();
            stream.Write(data, 0, data.Length);
            stream.Position = 0;
            BinaryFormatter bf = new BinaryFormatter();
            object obj = bf.Deserialize(stream);

            stream.Close();

            return obj;
        }
    }
}
