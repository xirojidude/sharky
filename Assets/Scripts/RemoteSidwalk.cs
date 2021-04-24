using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class RemoteSidwalk : UdonSharpBehaviour
{
    bool grab;
    public Texture2D sourceTex;
    public Rect sourceRect;
    public Camera camera;
    private VRCPlayerApi player;
    private float _ground = 0.0f;


    void Start()
    {
        grab=false;    
        player = Networking.LocalPlayer;
    }

    void Update()
    {
            Vector3 pos =  player.GetPosition();//this.transform.position;
            pos.y = this.transform.position.y;
            this.transform.position = pos;
    }

    public void setGround()
    {
        _ground = 1.0f; //groundHeight;
    }

    private void OnPostRender()
    {
           Vector3 pos =  player.GetPosition();//this.transform.position;
           pos.y = this.transform.position.y+_ground;
           this.transform.position = pos;
        if (grab)
        {
        //    Camera camera = Camera.main;
        //    camera.targetTexture = renderTexture;
        //    camera.Render();
            RenderTexture cameraTex = camera.activeTexture;

            int x = Mathf.FloorToInt(sourceRect.x);
            int y = Mathf.FloorToInt(sourceRect.y);
            int width = Mathf.FloorToInt(sourceRect.width);
            int height = Mathf.FloorToInt(sourceRect.height);

            Color[] pix = sourceTex.GetPixels(x, y, width, height);
//            Texture2D destTex = new Texture2D(width, height);
//            destTex.SetPixels(pix);
//            destTex.Apply();
            /*
            var m_Texture2D = new Texture2D(2, 2, TextureFormat.RGBA32, true);
            var mip0Data = m_Texture2D.GetPixelData<Color32>(1);


            // pixels in mip = 1 are filled with white color
            for (int i = 0; i < mip0Data.Length; i++)
            {
    //            mip0Data[i] = new Color32(255, 255, 255, 255);
            }

    //        m_Texture2D.Apply(false);
    */
        }

    }
}
