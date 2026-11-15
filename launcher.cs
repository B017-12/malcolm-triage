using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace MalcolmLauncher
{
    internal static class Program
    {
        private static Process _backendProc;
        private static Process _frontendProc;
        private static volatile bool _shuttingDown;

        private static int Main()
        {
            Console.Title = "Malcolm Clinical Triage — All-in-One Launcher";
            string baseDir = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory).TrimEnd('\\');
            string backendScript = Path.Combine(baseDir, "bedside_app.py");
            string frontendDir = Path.Combine(baseDir, "pixel-perfect-page-recraft");

            if (!File.Exists(backendScript) || !Directory.Exists(frontendDir))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[ERROR] MalcolmLauncher.exe must be placed in the root folder alongside bedside_app.py and pixel-perfect-page-recraft.");
                Console.ResetColor();
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey(true);
                return 1;
            }

            Console.CancelKeyPress += (sender, e) =>
            {
                e.Cancel = true;
                Shutdown();
            };
            AppDomain.CurrentDomain.ProcessExit += (sender, e) => Shutdown();

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("============================================================");
            Console.WriteLine("  MALCOLM CLINICAL TRIAGE SYSTEM — LAUNCHING");
            Console.WriteLine("============================================================");
            Console.ResetColor();

            try
            {
                if (IsHttpReady("http://127.0.0.1:5000/api/state"))
                {
                    Console.WriteLine("[Backend] Port 5000 is already active; reusing running API server.");
                }
                else
                {
                    Console.WriteLine("[Backend] Starting Flask RAG + Jarvis TTS API on port 5000...");
                    _backendProc = StartChild("python", "bedside_app.py", baseDir, "[API]");
                }

                if (IsHttpReady("http://localhost:8080/"))
                {
                    Console.WriteLine("[Frontend] Port 8080 is already active; reusing running React UI.");
                }
                else
                {
                    Console.WriteLine("[Frontend] Starting React Clinical UI on port 8080...");
                    _frontendProc = StartChild("cmd.exe", "/c npm --prefix \"pixel-perfect-page-recraft\" run dev", baseDir, "[UI]");
                }

                Console.WriteLine("[Launcher] Waiting for servers to become ready (up to 25s)...");
                bool backendUp = WaitForUrl("http://127.0.0.1:5000/api/state", 25000);
                bool frontendUp = WaitForUrl("http://localhost:8080/", 25000);

                string lanIp = GetLanIp();
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("============================================================");
                Console.WriteLine("  MALCOLM IS READY");
                Console.WriteLine("  Clinician UI:      http://localhost:8080/");
                Console.WriteLine("  Bedside UI (PC):   http://127.0.0.1:5000/");
                Console.WriteLine("  Bedside UI (Phone):http://" + lanIp + ":5000/");
                Console.WriteLine("============================================================");
                Console.ResetColor();
                Console.WriteLine("  Controls:");
                Console.WriteLine("    [R] Reset Demo State");
                Console.WriteLine("    [B] Open Bedside Patient UI in Browser");
                Console.WriteLine("    [C] Open Clinician UI in Browser");
                Console.WriteLine("    [Q] Quit and Stop All Servers");
                Console.WriteLine();

                if (!backendUp) Console.WriteLine("[WARN] Backend did not respond within 25s; check Python logs above.");
                if (!frontendUp) Console.WriteLine("[WARN] Frontend did not respond within 25s; check Vite logs above.");

                if (frontendUp)
                {
                    OpenBrowser("http://localhost:8080/");
                }

                while (!_shuttingDown)
                {
                    if (!Console.IsInputRedirected && Console.KeyAvailable)
                    {
                        ConsoleKeyInfo key = Console.ReadKey(true);
                        if (key.Key == ConsoleKey.Q)
                        {
                            break;
                        }
                        if (key.Key == ConsoleKey.R)
                        {
                            TriggerReset();
                        }
                        else if (key.Key == ConsoleKey.B)
                        {
                            OpenBrowser("http://127.0.0.1:5000/");
                        }
                        else if (key.Key == ConsoleKey.C)
                        {
                            OpenBrowser("http://localhost:8080/");
                        }
                    }
                    Thread.Sleep(150);
                }
            }
            finally
            {
                Shutdown();
            }

            return 0;
        }

        private static Process StartChild(string fileName, string arguments, string workingDir, string prefix)
        {
            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };

            var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine(prefix + " " + e.Data);
            };
            proc.ErrorDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine(prefix + " " + e.Data);
            };

            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            return proc;
        }

        private static bool WaitForUrl(string url, int timeoutMs)
        {
            var sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs && !_shuttingDown)
            {
                if (IsHttpReady(url)) return true;
                Thread.Sleep(400);
            }
            return false;
        }

        private static bool IsHttpReady(string url)
        {
            try
            {
                var req = (HttpWebRequest)WebRequest.Create(url);
                req.Method = "GET";
                req.Timeout = 1000;
                using (var resp = (HttpWebResponse)req.GetResponse())
                {
                    return (int)resp.StatusCode >= 200 && (int)resp.StatusCode < 400;
                }
            }
            catch
            {
                return false;
            }
        }

        private static void TriggerReset()
        {
            try
            {
                var req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:5000/api/reset");
                req.Method = "POST";
                req.ContentLength = 0;
                req.Timeout = 3000;
                using (var resp = (HttpWebResponse)req.GetResponse())
                {
                    Console.WriteLine("[Launcher] Demo state reset to idle (" + resp.StatusCode + ").");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("[Launcher] Reset failed: " + ex.Message);
            }
        }

        private static void OpenBrowser(string url)
        {
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = url,
                    UseShellExecute = true,
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine("[Launcher] Could not open browser automatically: " + ex.Message);
            }
        }

        private static string GetLanIp()
        {
            try
            {
                using (var sock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp))
                {
                    sock.Connect("10.255.255.255", 1);
                    var endPoint = sock.LocalEndPoint as IPEndPoint;
                    return endPoint != null ? endPoint.Address.ToString() : "127.0.0.1";
                }
            }
            catch
            {
                return "127.0.0.1";
            }
        }

        private static void Shutdown()
        {
            if (_shuttingDown) return;
            _shuttingDown = true;
            Console.WriteLine("\n[Launcher] Shutting down Malcolm servers...");
            KillTree(_frontendProc);
            KillTree(_backendProc);
        }

        private static void KillTree(Process proc)
        {
            if (proc == null) return;
            try
            {
                if (!proc.HasExited)
                {
                    var killer = Process.Start(new ProcessStartInfo
                    {
                        FileName = "taskkill",
                        Arguments = "/F /T /PID " + proc.Id,
                        UseShellExecute = false,
                        CreateNoWindow = true,
                    });
                    if (killer != null) killer.WaitForExit(3000);
                }
            }
            catch
            {
                // Best effort cleanup
            }
        }
    }
}
