Shader "Unlit/fire"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _rows ("Rows", float) = 4.0
        _cols ("Columns", float) = 5.0
        _speed ("Speed", float) = 100.0
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType"="Transparent" }
        LOD 100
         Blend SrcAlpha OneMinusSrcAlpha


        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
           // #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _rows,_cols,_speed;


            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float f = floor(_Time.y*_speed)%(_rows*_cols);
                float fy = floor(f/_cols) * 1/_rows;
                float fx = f%_cols * 1/_cols; 
                // sample the texture

                fixed4 col = tex2D(_MainTex, i.uv/float2(_cols,_rows)+float2(fx,fy));
                // apply fog
               // UNITY_APPLY_FOG(i.fogCoord, col);
               float alpha = (col.r + col.g + col.b)/3;
                return float4(col.rgb,alpha);
            }
            ENDCG
        }
    }
}
