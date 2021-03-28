
Shader "Skybox/BigBang"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }


         fixed4 frag (v2f v) : SV_Target
            {
                float2 F = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 col = float4(0,0,0,1); //tex2D(_MainTex, i.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin
                float2 sd = float2(rd.x / rd.z, rd.y / rd.z);

                float2 R = float2(800,450);//iResolution.xy;                                            
                for(float d,t = _Time.y*.1, i = 0. ; i > -1.; i -= .06 )              
                {   d = frac( i -3.*t );                                           
//                    float4 c = float4( ( F - R *.5 ) / R.y *d ,i,0 ) * 28.;             
                    float4 c = float4(  sd.x *d *.5,sd.y+i,0,0 ) * 28.;             
                    for (int j=0 ; j++ <27; )                                       
                        c.xzyw = abs( c / dot(c,c)                                  
                                -float4( 7.-.2*sin(t) , 6.3 , .7 , 1.-cos(t/.8))/7.); 
                   col += c * c.yzww  * (d-d*d)  / float4(3,5,1,1);                     
                }

                return col;
            }

            ENDCG
        }
    }
}



// for more clear code see https://www.shadertoy.com/view/WtjyzR

